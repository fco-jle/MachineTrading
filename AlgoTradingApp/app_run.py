import os
import sys
import time
import numpy as np
import pandas as pd
import datetime as dt

from sklearn.decomposition import PCA

from PyQt5.uic import loadUi
from PyQt5.QtCore import QDate
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow, QMessageBox

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyqtgraph.parametertree import ParameterTree, Parameter
from pyqtgraph.parametertree import types as ptypes

import pyqtgraph as pg
from pyqtgraph import configfile
from pyqtgraph.flowchart import Flowchart

from MachineTrading.Utils import io_utils
from MachineTrading.Instruments.fixed_rate import BTP
from MachineTrading.AlgoTradingApp.appmain import Ui_MainWindow


class AppBond:
    def __init__(self, isin):
        os.path.join(os.path.dirname(__file__), f"Data/Yields/{isin}.csv")
        bond_info = io_utils.read_json(os.path.join(os.path.dirname(__file__), f"Data/BondInfo/{isin}.json"))
        yield_data = io_utils.investing_yield_data(os.path.join(os.path.dirname(__file__), f"Data/Yields/{isin}.csv"))
        yield_data = yield_data.set_index('Date').resample(rule='D').first().fillna(method='ffill').reset_index()

        maturity = dt.datetime.strptime(bond_info["MaturityDate"], '%Y-%m-%d')
        issue = dt.datetime.strptime(bond_info["IssueDate"], '%Y-%m-%d')
        coupon = bond_info["CouponInterest"]
        self.instrument_id = isin
        self.bond = BTP(issue_date=issue, maturity_date=maturity, coupon=coupon)

        self._yield_data = yield_data.copy()
        self._yield_data_backup = yield_data.copy()

        self._price_data = self.compute_price_data()
        self._price_data_backup = self._price_data.copy()

        self.price = None
        self.yld = None
        self.convexity = None
        self.duration = None

    def compute_price_data(self):
        temp = self._yield_data.copy()
        for col in ['Close', 'Open', 'High', 'Low']:
            temp[col] = temp.apply(lambda row: self.bond.clean_price(row[col], row['Date'], errors='coerce'), axis=1)
        return temp

    def get_original_yield_data(self):
        return self._yield_data_backup

    def get_current_yield_data(self):
        return self._yield_data

    def set_current_yield_data(self, data):
        self._yield_data = data.copy()

    def get_original_price_data(self):
        return self._price_data_backup

    def get_current_price_data(self):
        return self._price_data

    def set_current_price_data(self, data):
        self._price_data = data.copy()

    def reset_yield_data(self):
        self._yield_data = self._yield_data_backup.copy()

    def reset_price_data(self):
        selt._price_data = self._price_data_backup.copy()

    def update_data(self, date):
        max_date_available = self.get_current_price_data()['Date'].iloc[-1].date()
        if date > max_date_available:
            print(f"Selected date not available, defaulting to max date in data: {max_date_available}")
            date = max_date_available

        self.bond.set_evaluation_date(date)
        self.price = self.get_current_price_data().set_index('Date').loc[str(date)]['Close']
        self.yld = self.get_current_yield_data().set_index('Date').loc[str(date)]['Close']
        self.convexity = self.bond.convexity(bond_yield=self.yld)
        self.duration = self.bond.duration_modified(bond_yield=self.yld)
        self.dv01 = self.bond.dv01(bond_yield=self.yld, bond_price=self.price, eval_date=date)
        self.years_to_maturity = self.bond.years_to_maturity(eval_date=date)


def random_color():
    return tuple(np.random.randint(0, 255, 3))


class InstrumentParam(ptypes.GroupParameter):
    def __init__(self, **kwds):
        name = kwds.get('IsinCode', 'Instrument')
        auto_ic = True if name == 'Instrument' else False
        defs = dict(
            name=name, autoIncrementName=auto_ic, renamable=True, removable=True, expanded=False, children=[
                dict(name='Plot', type='bool', value=True),
                dict(name='Notional', type='float', value=0.0, step=100, limits=[-1e12, 1e12]),
                dict(name='HedgeInstrument', type='bool', value=False),
                dict(name='Style', type='list', limits=['Solid', 'Dashed', 'Dotted'], value='Solid'),
                dict(name='Color', type='color', value=random_color(), default=(0, 0, 0)),
            ])
        ptypes.GroupParameter.__init__(self, **defs)
        self.restoreState(kwds, removeChildren=False)

    def instrument_name(self):
        return [self.name()]


class InstrumentsGroupParam(ptypes.GroupParameter):
    def __init__(self):
        ptypes.GroupParameter.__init__(self, name="Instruments", addText="Add New..", addList=[])
        self.set_options()

    def set_options(self):
        current_instruments = self.current_instrument_names()
        available_instruments = [x.split(".")[0] for x in os.listdir("Data/BondInfo") if '.json' in x]
        possible_instruments = [x for x in available_instruments if x not in current_instruments]
        self.setAddList(vals=[] + possible_instruments)

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


ptypes.registerParameterType('Instrument', InstrumentParam)


class Window(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.params = None
        self.instruments = {}
        self.latest_notionals = {}
        self.objectGroup = InstrumentsGroupParam()

        # Plot Layout
        self.setup_gui()
        self.setup_gui_2()

        # Tree
        self.setup_config()
        self.tree = ParameterTree(showHeader=False)

        # Other components:
        self.pca_data = {}

    def setup_gui_2(self):
        self.hedgeSplitter = QtWidgets.QSplitter()
        self.hedgeSplitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.ui.hedgeLayout.addWidget(self.hedgeSplitter)

        # Configure Object Time Series Plots
        self.hedgeInfoPlots = pg.GraphicsLayoutWidget()
        self.hedgeSplitter.addWidget(self.hedgeInfoPlots)
        self.hedgeSplitter.setSizes([int(self.width() * 0.6), int(self.width() * 0.4)])

        self.dv01_plot = self.hedgeInfoPlots.addPlot(row=0, col=0)
        self.duration_plot = self.hedgeInfoPlots.addPlot(row=1, col=0)
        self.convexity_plot = self.hedgeInfoPlots.addPlot(row=2, col=0)
        self.duration_plot.setXLink(self.dv01_plot)
        self.convexity_plot.setXLink(self.dv01_plot)

        self.dv01_plot.setTitle("<font size='5'> DV01 </font>")
        self.dv01_plot.setLabels(left="<font size='4'>DV01</font>", bottom="<font size='4'>Maturity</font>")
        self.duration_plot.setTitle("<font size='5'> Duration </font>")
        self.duration_plot.setLabels(left="<font size='4'>Duration</font>", bottom="<font size='4'>Maturity</font>")
        self.convexity_plot.setTitle("<font size='5'> Convexity </font>")
        self.convexity_plot.setLabels(left="<font size='4'>Convexity</font>", bottom="<font size='4'>Maturity</font>")

        self.dv01_plot.showGrid(x=False, y=True)
        self.duration_plot.showGrid(x=False, y=True)
        self.convexity_plot.showGrid(x=False, y=True)

    def setup_gui(self):
        self.splitter = QtWidgets.QSplitter()
        self.splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.ui.plotsLayout.addWidget(self.splitter)

        # Configure Object Time Series Plots
        self.time_series_plots = pg.GraphicsLayoutWidget()
        self.maturity_plot = pg.GraphicsLayoutWidget()
        self.splitter.addWidget(self.time_series_plots)
        self.splitter.addWidget(self.maturity_plot)
        self.splitter.setSizes([int(self.width() * 0.6), int(self.width() * 0.4)])

        # Config Left Part of Splitter
        self.yield_plot = self.time_series_plots.addPlot(row=0, col=0, axisItems={'bottom': pg.DateAxisItem()})
        self.price_plot = self.time_series_plots.addPlot(row=1, col=0, axisItems={'bottom': pg.DateAxisItem()})
        self.price_plot.setXLink(self.yield_plot)
        self.yield_plot.showGrid(x=False, y=True)
        self.price_plot.showGrid(x=False, y=True)
        self.yield_plot.setTitle("<font size='5'> Selected Instruments Yield</font>")
        self.yield_plot.setLabels(left="<font size='4'>Yield</font>", bottom="<font size='4'>Datetime</font>")
        self.price_plot.setTitle("<font size='5'> Selected Instruments Price</font>")
        self.price_plot.setLabels(left="<font size='4'>Price</font>", bottom="<font size='4'>Datetime</font>")

        # Config Right Part of Splitter
        self.yield_curve_plot = self.maturity_plot.addPlot(row=0, col=0)
        self.volatility_plot = self.maturity_plot.addPlot(row=1, col=0)
        self.pca_plot = self.maturity_plot.addPlot(row=2, col=0)
        self.volatility_plot.setXLink(self.yield_curve_plot)
        self.pca_plot.setXLink(self.yield_curve_plot)
        self.yield_curve_plot.showGrid(x=False, y=True)
        self.volatility_plot.showGrid(x=False, y=True)
        self.pca_plot.showGrid(x=False, y=True)
        self.yield_curve_plot.setTitle("<font size='5'> Yield Curve </font>")
        self.yield_curve_plot.setLabels(left="<font size='4'>Yield</font>", bottom="<font size='4'>Maturity</font>")
        self.volatility_plot.setTitle("<font size='5'> Volatility </font>")
        self.volatility_plot.setLabels(left="<font size='4'>Volatility</font>", bottom="<font size='4'>Maturity</font>")
        self.pca_plot.setTitle("<font size='5'> PCA </font>")
        self.pca_plot.setLabels(left="<font size='4'>PCA</font>", bottom="<font size='4'>Maturity</font>")

    def setup_config(self):
        self.tree = ParameterTree(showHeader=False)
        self.params = Parameter.create(name='params', type='group', children=[
            dict(name='Load Preset', type='list', limits=[]),
            dict(name='Save', type='action'),
            dict(name='Load', type='action'),
            Parameter.create(name='Data Params', type='group', children=[
                dict(name='Candles Window', type='str', value="D"),
                dict(name='EMA Window', type='int', value=0, step=1, limits=[0, None]),
                dict(name='Volatility Type', type='list',
                     limits=["Garman Klass", "Standard Dev"], value="Garman Klass"),
                dict(name='PCA Components', type='int', value=3),
                dict(name='Yield Curve Date', type='calendar', value=None, expanded=False),
            ]),
            self.objectGroup,
            Parameter.create(name='Hedge Options', type='group', children=[
                dict(name='Hedge Type', type='list',
                     limits=["DV01", "DV01-Convexity", "YieldVariance", "PriceVariance"], value="DV01"),
                dict(name='Notional Multiply', type='bool', value=False)
            ])
        ])
        self.tree.setParameters(self.params, showTop=False)
        self.params.param('Save').sigActivated.connect(self.save)
        self.params.param('Load').sigActivated.connect(self.load)
        self.params.param('Load Preset').sigValueChanged.connect(self.load_preset)
        self.params.param('Instruments').sigChildAdded.connect(self.instrument_add)
        self.params.param('Instruments').sigChildRemoved.connect(self.instrument_delete)
        self.params.param('Data Params').param('Candles Window').sigValueChanged.connect(self.update_candles)
        self.params.param('Data Params').param('EMA Window').sigValueChanged.connect(self.update_candles)
        self.params.param('Data Params').param('Yield Curve Date').sigValueChanged.connect(self.update_maturity_plot)
        self.params.sigTreeStateChanged.connect(self.params_change_manager)

        preset_dir = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'presets')
        if os.path.exists(preset_dir):
            presets = [os.path.splitext(p)[0] for p in os.listdir(preset_dir)]
            self.params.param('Load Preset').setLimits([''] + presets)
        self.ui.instrumentsLayout.addWidget(self.tree)

    def instrument_add(self, child, idx):
        is_generic = idx.name().startswith('Instrument')
        if not is_generic:
            new_instrument_isin = idx.name().split(' ')[0]
            bond = AppBond(new_instrument_isin)
            self.instruments[new_instrument_isin] = bond
        self.update_price_plot()
        self.params.param('Instruments').param(
            new_instrument_isin).param('Notional').sigValueChanged.connect(self.apply_hedge)
        self.params.param('Instruments').param(
            new_instrument_isin).param('HedgeInstrument').sigValueChanged.connect(self.apply_hedge)

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

    def update_price_plot(self):
        if not self.yield_plot:
            return

        self.yield_plot.clear()
        self.price_plot.clear()
        for isin in self.instruments.keys():
            try:
                isin_params = self.params.param('Instruments').param(isin)
            except KeyError:
                continue

            # Check if should be plotted:
            if not isin_params['Plot']:
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
            prc_data = app_bond.get_current_price_data()

            x = yld_data['Date'].values.astype(np.int64) // 10 ** 9
            y = yld_data['Close'].values

            x2 = prc_data['Date'].values.astype(np.int64) // 10 ** 9
            y2 = prc_data['Close'].values
            self.yield_plot.plot(x, y, pen=pen)
            self.price_plot.plot(x2, y2, pen=pen)

    def get_selected_date(self):
        date_param = self.params.param('Data Params').param("Yield Curve Date").value()
        date = dt.date(date_param.year(), date_param.month(), date_param.day())
        return date

    def update_maturity_plot(self):
        self.yield_curve_plot.clear()
        self.volatility_plot.clear()
        self.pca_plot.clear()

        # Pen Creation
        pen = pg.mkPen(color=(206, 241, 229))

        volatility_x = []
        volatility_y = []
        yields_y = []

        for i, isin in enumerate(self.instruments.keys()):
            app_bond = self.instruments[isin]
            assert isinstance(app_bond, AppBond)
            try:
                isin_params = self.params.param('Instruments').param(isin)
            except KeyError:
                continue
            if not isin_params['Plot']:
                continue

            prc_data = app_bond.get_current_price_data()
            yld_data = app_bond.get_current_yield_data()

            try:
                date = self.get_selected_date()
                current_yld = yld_data.set_index('Date').loc[str(date)]['Close']
            except KeyError:
                current_yld = yld_data.set_index('Date').iloc[-1]['Close']

            yields_y.append(current_yld)
            ytm = app_bond.bond.years_to_maturity()
            vola = prc_data['Close'].std()
            volatility_x.append(ytm)
            volatility_y.append(vola)

        volatility_x = np.array(volatility_x)
        volatility_y = np.array(volatility_y)
        yields_y = np.array(yields_y)

        # Sort Arrays:
        sorted_idx = np.argsort(volatility_x)
        self.yield_curve_plot.plot(volatility_x[sorted_idx], yields_y[sorted_idx], symbol='o', pen=pen)
        self.volatility_plot.plot(volatility_x[sorted_idx], volatility_y[sorted_idx], symbol='o', pen=pen)

        if 'Components' in self.pca_data.keys():
            n_comps = self.pca_data['Components'].shape[0]
            pens = [pg.mkPen(color=random_color()) for _ in range(n_comps)]
            for i in range(n_comps):
                self.pca_plot.plot(self.pca_data['X'], self.pca_data['Components'][i], pen=pens[i])

    def update_candles(self):
        for instrument_id, bond in self.instruments.items():
            assert isinstance(bond, AppBond)
            data = [bond.get_original_yield_data().copy(), bond.get_original_price_data().copy()]
            new_data = []
            for df in data:
                candles_w = self.params.param('Data Params').param("Candles Window").value()
                ema_w = self.params.param('Data Params').param("EMA Window").value()

                df = df.set_index('Date').resample(rule=candles_w).last()
                df = df.dropna(thresh=4)
                df = df.reset_index()

                # If EMA Window exists, transform:
                if ema_w >= 1:
                    df = df.set_index('Date').ewm(span=ema_w).mean().reset_index()
                new_data.append(df)

            bond.set_current_yield_data(new_data[0])
            bond.set_current_price_data(new_data[1])

        self.params_change_manager()

    def update_pca_computation(self):
        pca_df = []
        maturities = []
        isins = []
        for i, isin in enumerate(self.instruments.keys()):
            app_bond = self.instruments[isin]
            assert isinstance(app_bond, AppBond)
            try:
                isin_params = self.params.param('Instruments').param(isin)
            except KeyError:
                continue
            if not isin_params['Plot']:
                continue

            ytm = app_bond.bond.years_to_maturity()
            yld_data = app_bond.get_current_yield_data()
            close_yld = yld_data.set_index('Date')[['Close']]
            close_yld.columns = [app_bond.instrument_id]
            pca_df.append(close_yld)
            maturities.append(ytm)
            isins.append(app_bond.instrument_id)

        try:
            pca_df = pd.concat(pca_df, axis=1)
        except ValueError:
            self.pca_data = {}
            return

        sorted_indexes = np.argsort(maturities)
        sorted_instruments = [isins[x] for x in sorted_indexes]
        sorted_maturities = np.array([maturities[x] for x in sorted_indexes])
        sorted_df = pca_df[sorted_instruments].copy()
        sorted_df = sorted_df.fillna(method='ffill')

        n_comps = self.params.param('Data Params').param('PCA Components').value()
        pca = PCA(n_components=n_comps)
        try:
            pca.fit(sorted_df)
            self.pca_data = {'X': sorted_maturities,
                             'Components': pca.components_}
        except ValueError:
            self.pca_data = {}

    def update_hedge_plot(self):
        maturities = []
        dv01s = []
        durations = []
        convexities = []
        isins = []
        notionals = []
        multiply_by_notionals = self.params.param("Hedge Options").param('Notional Multiply').value()
        for i, isin in enumerate(self.instruments.keys()):
            app_bond = self.instruments[isin]
            assert isinstance(app_bond, AppBond)
            try:
                isin_params = self.params.param('Instruments').param(isin)
            except KeyError:
                continue
            if not isin_params['Plot']:
                continue

            date = self.get_selected_date()
            app_bond.update_data(date=date)
            maturities.append(app_bond.years_to_maturity)
            dv01s.append(app_bond.dv01)
            durations.append(app_bond.duration)
            convexities.append(app_bond.convexity)
            isins.append(app_bond.instrument_id)
            notionals.append(isin_params['Notional'])

        sorted_indexes = np.argsort(maturities)
        sorted_maturities = np.array([maturities[x] for x in sorted_indexes])
        sorted_notionals = np.array([notionals[x] for x in sorted_indexes])

        sorted_dv01s = np.array([dv01s[x] for x in sorted_indexes])
        sorted_durations = np.array([durations[x] for x in sorted_indexes])
        sorted_convexities = np.array([convexities[x] for x in sorted_indexes])

        if multiply_by_notionals:
            sorted_dv01s *= sorted_notionals
            sorted_durations *= sorted_notionals
            sorted_convexities *= sorted_notionals

        for plot in [self.dv01_plot, self.duration_plot, self.convexity_plot]:
            plot.clear()

        bargraph1 = pg.BarGraphItem(x=sorted_maturities, height=sorted_dv01s, width=0.3, brush='#43bce8')
        bargraph2 = pg.BarGraphItem(x=sorted_maturities, height=sorted_durations, width=0.3, brush='#43bce8')
        bargraph3 = pg.BarGraphItem(x=sorted_maturities, height=sorted_convexities, width=0.3, brush='#43bce8')
        self.dv01_plot.addItem(bargraph1)
        self.duration_plot.addItem(bargraph2)
        self.convexity_plot.addItem(bargraph3)

    def apply_hedge(self):
        if self.notionals_have_changed():
            hedge = np.sum(list(self.latest_notionals.values()))
            print("Hedge Calculated! : ", hedge)

    def notionals_have_changed(self):
        notionals = {}
        for i, isin in enumerate(self.instruments.keys()):
            app_bond = self.instruments[isin]
            assert isinstance(app_bond, AppBond)
            try:
                isin_params = self.params.param('Instruments').param(isin)
            except KeyError:
                continue

            notionals[app_bond.instrument_id] = isin_params['Notional']
        if notionals != self.latest_notionals:
            self.latest_notionals = notionals
            return True
        else:
            return False

    def params_change_manager(self):
        self.update_pca_computation()
        self.update_price_plot()
        self.update_maturity_plot()
        self.update_hedge_plot()


if __name__ == "__main__":
    from pyqtgraph.examples.relativity import relativity

    # from pyqtgraph.examples import parametertree
    test = False
    if test:
        import pyqtgraph.examples

        pyqtgraph.examples.run()

    else:
        app = QApplication(sys.argv)
        win = Window()
        win.show()
        sys.exit(app.exec())
