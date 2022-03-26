import os
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow, QMessageBox
from PyQt5.uic import loadUi

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyqtgraph.parametertree import ParameterTree, Parameter
from pyqtgraph.parametertree import types as ptypes
from AlgoTradingApp.appmain import Ui_MainWindow
from pyqtgraph.flowchart import Flowchart
import pyqtgraph as pg
from pyqtgraph import configfile
from Instruments.fixed_rate import FixedRateBond
import datetime as dt
from Utils import io_utils
import time


class AppBond:
    def __init__(self, isin):
        bond_info = io_utils.read_json(f"Data/BondInfo/{isin}.json")
        yield_data = io_utils.investing_yield_data(f"Data/Yields/{isin}.csv")
        maturity = dt.datetime.strptime(bond_info["MaturityDate"], '%Y-%m-%d')
        issue = dt.datetime.strptime(bond_info["IssueDate"], '%Y-%m-%d')
        coupon = bond_info["CouponInterest"]
        self.bond = FixedRateBond(issue_date=issue, maturity_date=maturity, coupon=coupon, settlement_days=2)
        self._yield_data = yield_data.copy()
        self._yield_data_backup = yield_data.copy()

    def get_original_yield_data(self):
        return self._yield_data_backup

    def get_current_yield_data(self):
        return self._yield_data

    def set_current_yield_data(self, data):
        self._yield_data = data.copy()

    def reset_yield_data(self):
        self._yield_data = self._yield_data_backup.copy()


class InstrumentsGroupParam(ptypes.GroupParameter):
    def __init__(self):
        ptypes.GroupParameter.__init__(self, name="Instruments", addText="Add New..", addList=['Instrument'])
        self.set_options()

    def set_options(self):
        current_instruments = self.current_instrument_names()
        available_instruments = [x.split(".")[0] for x in os.listdir("Data/BondInfo")]
        possible_instruments = [x for x in available_instruments if x not in current_instruments]
        self.setAddList(vals=['Instrument'] + possible_instruments)

    def current_instrument_names(self):
        return [x.instrument_name()[0] for x in self.childs]

    def addNew(self, typ=None):
        if typ == 'Instrument':
            self.addChild(InstrumentParam())
        else:
            self.addChild(InstrumentParam(**{'IsinCode': typ}))
        self.set_options()

    def removeChild(self, child):
        ptypes.GroupParameter.removeChild(self, child)
        self.set_options()


class InstrumentParam(ptypes.GroupParameter):
    def __init__(self, **kwds):
        name = kwds.get('IsinCode', 'Instrument')
        auto_ic = True if name == 'Instrument' else False
        defs = dict(
            name=name, autoIncrementName=auto_ic, renamable=True, removable=True, children=[
                dict(name='Plots', type='bool', value=True),
                dict(name='Notional', type='float', value=0.0, step=100, limits=[-1e12, 1e12]),
                # AccelerationGroup(),
                dict(name='Style', type='list', limits=['Solid', 'Dashed', 'Dotted'], value='Solid'),
                dict(name='Color', type='color', value=self.random_color(), default=(0, 0, 0)),
            ])
        ptypes.GroupParameter.__init__(self, **defs)
        self.restoreState(kwds, removeChildren=False)

    def instrument_name(self):
        return [self.name()]

    def random_color(self):
        return tuple(np.random.randint(0, 255, 3))


ptypes.registerParameterType('Instrument', InstrumentParam)


class AccelerationGroup(ptypes.GroupParameter):
    def __init__(self, **kwds):
        defs = dict(name="Acceleration", addText="Add Command..")
        ptypes.GroupParameter.__init__(self, **defs)
        self.restoreState(kwds, removeChildren=False)

    def addNew(self, typ=None):
        next_time = 0.0
        if self.hasChildren():
            next_time = self.children()[-1]['Proper Time'] + 1
        self.addChild(
            Parameter.create(name='Command', autoIncrementName=True, type=None, renamable=True, removable=True,
                             children=[
                                 dict(name='Proper Time', type='float', value=next_time),
                                 dict(name='Acceleration', type='float', value=0.0, step=0.1),
                             ]))

    def generate(self):
        prog = []
        for cmd in self:
            prog.append((cmd['Proper Time'], cmd['Acceleration']))
        return prog


class Window(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.params = None
        self.instruments = {}
        self.objectGroup = InstrumentsGroupParam()

        # Plot Layout
        # self.splitter = QtWidgets.QSplitter()
        # self.splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)

        self.graphic_layout_widget = pg.GraphicsLayoutWidget()
        self.ui.plotsLayout.addWidget(self.graphic_layout_widget)

        self.price_plot = self.graphic_layout_widget.addPlot(axisItems={'bottom': pg.DateAxisItem()})
        self.price_plot.showGrid(x=True, y=True)

        # Tree
        self.setup_config()
        self.tree = ParameterTree(showHeader=False)

    def setup_config(self):
        self.tree = ParameterTree(showHeader=False)
        self.params = Parameter.create(name='params', type='group', children=[
            dict(name='Load Preset', type='list', limits=[]),
            dict(name='Candles Window', type='str', value="D"),
            dict(name='EMA Window', type='int', value=0, step=1, limits=[0, None]),
            dict(name='Volatility Type', type='list', limits=[]),
            dict(name='Price Volatility', type='bool', value=True),
            dict(name='Yield Volatility', type='bool', value=True),
            # dict(name='Animation Speed', type='float', value=1.0, dec=True, step=0.1, limits=[0.0001, None]),
            dict(name='Save', type='action'),
            dict(name='Load', type='action'),
            self.objectGroup,
        ])
        self.tree.setParameters(self.params, showTop=False)
        self.params.param('Save').sigActivated.connect(self.save)
        self.params.param('Load').sigActivated.connect(self.load)
        self.params.param('Load Preset').sigValueChanged.connect(self.load_preset)
        self.params.param('Instruments').sigChildAdded.connect(self.instrument_add)
        self.params.param('Instruments').sigChildRemoved.connect(self.instrument_delete)

        self.params.param('Candles Window').sigValueChanged.connect(self.update_candles)
        self.params.param('EMA Window').sigValueChanged.connect(self.update_candles)

        self.params.sigTreeStateChanged.connect(self.params_change_manager)

        preset_dir = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'presets')
        if os.path.exists(preset_dir):
            presets = [os.path.splitext(p)[0] for p in os.listdir(preset_dir)]
            self.params.param('Load Preset').setLimits([''] + presets)
        self.ui.instrumentsLayout.addWidget(self.tree)

    def instrument_add(self, child, idx):
        is_generic = idx.name().startswith('Instrument')
        if not is_generic:
            new_instrument_isin = idx.name()
            bond = AppBond(new_instrument_isin)
            self.instruments[new_instrument_isin] = bond
        # self.update_price_plot()

    def instrument_delete(self, child: InstrumentsGroupParam):
        to_delete = list(set(self.instruments.keys()) - set(child.current_instrument_names()))
        if len(to_delete) > 0:
            del self.instruments[to_delete[0]]

    def save(self):
        filename = pg.QtWidgets.QFileDialog.getSaveFileName(self, "Save State..", "untitled.cfg",
                                                            "Config Files (*.cfg)")
        if isinstance(filename, tuple):
            filename = filename[0]  # Qt4/5 API difference
        if filename == '':
            return
        state = self.params.saveState()
        configfile.writeConfigFile(state, str(filename))

    def load(self):
        filename = pg.QtWidgets.QFileDialog.getOpenFileName(self, "Save State..", "", "Config Files (*.cfg)")
        if isinstance(filename, tuple):
            filename = filename[0]  # Qt4/5 API difference
        if filename == '':
            return
        state = configfile.readConfigFile(str(filename))
        self.load_state(state)

    def load_preset(self, param, preset):
        if preset == '':
            return
        path = os.path.abspath(os.path.dirname(__file__))
        fn = os.path.join(path, 'presets', preset + ".cfg")
        state = configfile.readConfigFile(fn)
        self.load_state(state)

    def load_state(self, state):
        if 'Load Preset..' in state['children']:
            del state['children']['Load Preset..']['limits']
            del state['children']['Load Preset..']['value']
        self.params.param('Instruments').clearChildren()
        self.params.restoreState(state, removeChildren=False)
        # self.recalculate()

    def update_price_plot(self):
        if not self.price_plot:
            return
        self.price_plot.clear()
        for isin in self.instruments.keys():
            try:
                isin_params = self.params.param('Instruments').param(isin)
            except KeyError:
                continue

            # Check if should be plotted:
            if not isin_params['Plots']:
                continue

            # Pen Creation
            style_dict = {'Solid': None,
                          'Dotted': pg.QtCore.Qt.DotLine,
                          'Dashed': pg.QtCore.Qt.DashLine}

            rgb = isin_params['Color'].getRgb()
            pen = pg.mkPen(color=rgb[0:3], style=style_dict[isin_params['Style']])

            app_bond = self.instruments[isin]
            assert isinstance(app_bond, AppBond)
            yld_data = app_bond.get_current_yield_data()
            x = yld_data['Date'].values.astype(np.int64) // 10 ** 9
            y = yld_data['Close'].values
            self.price_plot.plot(x, y, pen=pen)

    def update_candles(self):
        for instrument_id, bond in self.instruments.items():
            assert isinstance(bond, AppBond)
            df = bond.get_original_yield_data().copy()

            # Create candles according to selected temporal window:
            df = df.set_index('Date').resample(rule=self.params['Candles Window']).last()
            df = df.dropna(thresh=4)
            df = df.reset_index()

            # If EMA Window exists, transform:
            if self.params['EMA Window'] >= 1:
                df = df.set_index('Date').ewm(span=self.params['EMA Window']).mean().reset_index()

            bond.set_current_yield_data(df)
        self.update_price_plot()

    def params_change_manager(self):
        self.update_price_plot()


if __name__ == "__main__":
    from pyqtgraph.examples.relativity import relativity

    test = False

    if test:
        import pyqtgraph.examples

        pyqtgraph.examples.run()

    else:
        app = QApplication(sys.argv)
        win = Window()
        win.show()
        sys.exit(app.exec())