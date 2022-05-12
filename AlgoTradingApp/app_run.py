import os
import sys
import time
import numpy as np
import pandas as pd
import datetime as dt
from scipy.optimize import minimize
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
import pyqtgraph.opengl as gl
from scipy.optimize import curve_fit

from MachineTrading.Utils import io_utils
from MachineTrading.Instruments.fixed_rate import BTP
from MachineTrading.AlgoTradingApp.appmain import Ui_MainWindow
from MachineTrading.TSA.volatility import garman_klass, standard_dev, close_to_close, parkinson, garman_klass_yang_zhang
from MachineTrading.yield_curve_functions import nelson_siegel


def random_color():
    return tuple(np.random.randint(0, 255, 3))


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
        self.current_data_date = None

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

    def update_data(self, date, log=False):
        if date == self.current_data_date:
            if log:
                AppLogger.print(f"Calling - update_data for isin {self.instrument_id} - Data is up to date - Skipping")
            return

        if log:
            AppLogger.print(f"Calling - update_data for isin {self.instrument_id}")
        max_date_available = self.get_current_price_data()['Date'].iloc[-1].date()
        if date > max_date_available:
            print(f"Selected date not available, defaulting to max date in data: {max_date_available}")
            date = max_date_available
        self.bond.set_evaluation_date(date)
        self.price = self.get_current_price_data().set_index('Date').loc[str(date)]['Close']
        self.yld = self.get_current_yield_data().set_index('Date').loc[str(date)]['Close']
        self.convexity = self.bond.convexity(bond_yield=self.yld)
        self.duration = self.bond.duration_modified(bond_yield=self.yld)
        self.dv01 = -self.bond.bpv(price=self.price, eval_date=date)*100
        self.years_to_maturity = self.bond.years_to_maturity(eval_date=date)
        self.current_data_date = date

    def compute_volatility(self, vola_type):
        if vola_type == 'Garman Klass':
            vola = garman_klass(self.get_current_price_data())
        elif vola_type == 'Standard Dev':
            vola = standard_dev(self.get_current_price_data())
        elif vola_type == 'Close To Close':
            vola = close_to_close(self.get_current_price_data())
        elif vola_type == 'Parkinson':
            vola = parkinson(self.get_current_price_data())
        elif vola_type == "Garman Klass Yang Zhang":
            vola = garman_klass_yang_zhang(self.get_current_price_data())
        else:
            print("WTF")
            vola = 0.0
        return vola


class InstrumentParam(ptypes.GroupParameter):
    def __init__(self, **kwds):
        name = kwds.get('IsinCode', 'Instrument')
        auto_ic = True if name == 'Instrument' else False
        defs = dict(
            name=name, autoIncrementName=auto_ic, renamable=True, removable=True, expanded=False, children=[
                dict(name='Enabled', type='bool', value=True),
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


class AppLogger:
    verbosity = 1

    @classmethod
    def print(cls, text):
        if cls.verbosity >= 1:
            print(text)


class Window(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.params = None
        self.instruments = {}
        self.latest_notionals = {}
        self.selected_date = dt.date
        self.current_hedge_method = None
        self.current_hedge = {}
        self.current_hedge_instruments = {}
        self.objectGroup = InstrumentsGroupParam()

        # Hedge Functions:
        self.hedge_funcs = {'Duration': self.duration_hedge,
                            'Duration-Convexity': self.duration_convexity_hedge,
                            'Duration-Convexity-DV01': self.duration_convexity_dv01_hedge,
                            'YieldVariance': self.duration_hedge,
                            'PriceVariance': self.variance_minimization_hedge,
                            'Quadratic': self.quadratic_hedge,
                            'Cubic':self.cubic_hedge,
                            }

        # Plot Layout
        self.setup_gui()
        self.setup_gui_2()

        # Tree
        self.setup_config()
        self.tree = ParameterTree(showHeader=False)

        # Other components:
        self.is_enabled = []
        self.pca_data = {}
        self.maturities = np.zeros(1)
        self.isins = np.zeros(1)
        self.is_hedge = np.zeros(1)
        self.prices = np.zeros(1)
        self.yields = np.zeros(1)
        self.volatilities = np.zeros(1)
        self.notionals = np.zeros(1)
        self.dv01s = np.zeros(1)
        self.durations = np.zeros(1)
        self.convexities = np.zeros(1)
        self.yield_series = None
        self.price_series = None

    def update_yield_surface_plot(self):
        self.yield_surface_plot.clear()
        if len(self.maturities) < 4:
            return

        w = self.yield_surface_plot
        w.setCameraPosition(distance=2)

        g = gl.GLGridItem()
        g.scale(0.5, 0.5, 1)
        g.setDepthValue(10)  # draw grid after surfaces since they may be translucent
        w.addItem(g)

        selected_date = self.get_selected_date()

        try:
            selected_date_idx = np.where(self.yield_series.index.date==selected_date)[0][0]
        except IndexError:
            selected_date_idx = len(self.yield_series)-1

        y = self.yield_series.index.values.astype(np.int64) // 10 ** 9
        y = (y-np.min(y)) / (np.max(y)-np.min(y))

        x = self.maturities
        x = (x - np.min(x)) / (np.max(x) - np.min(x))

        z = self.yield_series.values / np.max(self.yield_series.values)
        n = len(y)

        for i in range(n):
            yi = y[i]
            zi = z[i]
            pts = np.column_stack([x, np.ones(len(x))*yi, zi])
            color = pg.mkColor('r') if i == selected_date_idx else pg.mkColor(0.5)
            width = 5 if i==selected_date_idx else 1
            plt = gl.GLLinePlotItem(pos=pts, color=color, width=width, antialias=True)
            w.addItem(plt)


        # shader = "normalColor"
        # shader = 'heightColor'
        # # shader = 'shaded'
        # p2 = gl.GLSurfacePlotItem(x=x, y=y, z=z, smooth=True, shader=shader, glOptions='opaque')
        # w.addItem(p2)

    # ------------------------ SETUP -----------------------------
    def setup_gui(self):
        self.splitter = QtWidgets.QSplitter()
        self.splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.ui.plotsLayout.addWidget(self.splitter)

        self.splitter_vertical = QtWidgets.QSplitter()
        self.splitter_vertical.setOrientation(QtCore.Qt.Orientation.Vertical)

        # Configure Object Time Series Plots
        self.time_series_plots = pg.GraphicsLayoutWidget()
        self.maturity_plot = pg.GraphicsLayoutWidget()
        self.correlation_plot = pg.GraphicsLayoutWidget()
        self.yield_surface_plot = gl.GLViewWidget()

        self.splitter.addWidget(self.time_series_plots)
        self.splitter.addWidget(self.splitter_vertical)
        self.splitter.addWidget(self.maturity_plot)

        self.splitter_vertical.addWidget(self.yield_surface_plot)
        self.splitter_vertical.addWidget(self.correlation_plot)

        self.splitter.setSizes([int(self.width() * 0.50), int(self.width() * 0.25), int(self.width() * 0.25)])
        self.splitter_vertical.setSizes([int(self.height() * 0.33), int(self.height() * 0.667)])

        # Config Correlation Plot
        self.yield_corr_plot = self.correlation_plot.addPlot(row=0, col=0)
        self.price_corr_plot = self.correlation_plot.addPlot(row=1, col=0)
        self.yield_corr_plot.setTitle("<font size='5'> Yield Correlation Matrix </font>")
        self.price_corr_plot.setTitle("<font size='5'> Price Correlation Matrix </font>")

        # Config Left Part of Splitter
        self.yield_plot = self.time_series_plots.addPlot(row=0, col=0, axisItems={'bottom': pg.DateAxisItem()})
        self.price_plot = self.time_series_plots.addPlot(row=1, col=0, axisItems={'bottom': pg.DateAxisItem()})
        self.price_plot.setXLink(self.yield_plot)
        self.yield_plot.showGrid(x=False, y=True)
        self.price_plot.showGrid(x=False, y=True)
        self.yield_plot.setTitle("<font size='5'> Selected Instruments Yield</font>")
        self.yield_plot.setLabels(left="<font size='4'>Yield (%)</font>", bottom="<font size='4'>Datetime</font>")
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
        self.yield_curve_plot.setLabels(left="<font size='4'>Yield (%)</font>", bottom="<font size='4'>Maturity</font>")
        self.volatility_plot.setTitle("<font size='5'> Volatility </font>")
        self.volatility_plot.setLabels(left="<font size='4'>Volatility</font>", bottom="<font size='4'>Maturity</font>")
        self.pca_plot.setTitle("<font size='5'> PCA </font>")
        self.pca_plot.setLabels(left="<font size='4'>PCA</font>", bottom="<font size='4'>Maturity</font>")

        self.pca_plot.addLegend()

    def setup_gui_2(self):
        self.hedgeSplitter = QtWidgets.QSplitter()
        self.hedgeSplitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.ui.hedgeLayout.addWidget(self.hedgeSplitter)

        # Configure Object Time Series Plots
        self.hedgeInfoData = pg.LayoutWidget()
        self.hedgeSplitter.addWidget(self.hedgeInfoData)

        self.hedge_data_table = pg.TableWidget()
        self.hedge_data_table.show()
        self.hedge_data_table.resize(500, 500)
        self.hedge_data_table.setWindowTitle('pyqtgraph example: TableWidget')
        self.hedgeInfoData.addWidget(self.hedge_data_table)

        self.hedgeInfoPlots = pg.GraphicsLayoutWidget()
        self.hedgeSplitter.addWidget(self.hedgeInfoPlots)

        self.hedgeSplitter.setSizes([int(self.width() * 0.25), int(self.width() * 0.75)])

        self.dv01_plot = self.hedgeInfoPlots.addPlot(row=0, col=0)
        self.duration_plot = self.hedgeInfoPlots.addPlot(row=1, col=0)
        self.convexity_plot = self.hedgeInfoPlots.addPlot(row=2, col=0)
        self.hedge_daily_pl_plot = self.hedgeInfoPlots.addPlot(row=3, col=0, axisItems={'bottom': pg.DateAxisItem()})

        self.hedge_daily_pl_plot.addLegend()

        self.duration_plot.setXLink(self.dv01_plot)
        self.convexity_plot.setXLink(self.dv01_plot)

        self.dv01_plot.setTitle("<font size='5'> DV01 </font>")
        self.dv01_plot.setLabels(left="<font size='4'>DV01</font>", bottom="<font size='4'>Maturity</font>")
        self.duration_plot.setTitle("<font size='5'> Duration </font>")
        self.duration_plot.setLabels(left="<font size='4'>Duration</font>", bottom="<font size='4'>Maturity</font>")
        self.convexity_plot.setTitle("<font size='5'> Convexity </font>")
        self.convexity_plot.setLabels(left="<font size='4'>Convexity</font>", bottom="<font size='4'>Maturity</font>")
        self.hedge_daily_pl_plot.setTitle("<font size='5'> Daily P&L Variations </font>")
        self.hedge_daily_pl_plot.setLabels(left="<font size='4'>P&L</font>", bottom="<font size='4'>Date</font>")

        self.dv01_plot.showGrid(x=False, y=True)
        self.duration_plot.showGrid(x=False, y=True)
        self.convexity_plot.showGrid(x=False, y=True)
        self.hedge_daily_pl_plot.showGrid(x=False, y=True)

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
                     limits=["Standard Dev", "Close To Close", "Parkinson", "Garman Klass", "Garman Klass Yang Zhang"],
                     value="Garman Klass"),
                dict(name='PCA Components', type='int', value=3),
                dict(name='Date', type='calendar', value=None, expanded=False),
            ]),
            self.objectGroup,
            Parameter.create(name='Hedge Options', type='group', children=[
                dict(name='Hedge Type', type='list',
                     limits=list(self.hedge_funcs.keys()),
                     # limits=["Duration", "Duration-Convexity", "YieldVariance", "PriceVariance", "Quadratic"],
                     value="Duration"),
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
        self.params.param('Data Params').param('Date').sigValueChanged.connect(self.params_change_manager)
        self.params.param('Data Params').param('PCA Components').sigValueChanged.connect(self.update_pca_computation)
        self.params.param('Data Params').param('Volatility Type').sigValueChanged.connect(self.update_volatility)
        self.params.param('Hedge Options').param('Hedge Type').sigValueChanged.connect(self.apply_hedge)
        self.params.param("Hedge Options").param('Notional Multiply').sigValueChanged.connect(self.update_hedge_tab)

        preset_dir = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), 'presets')
        if os.path.exists(preset_dir):
            presets = [os.path.splitext(p)[0] for p in os.listdir(preset_dir)]
            self.params.param('Load Preset').setLimits([''] + presets)
        self.ui.instrumentsLayout.addWidget(self.tree)

    # ----------------- ADDING / DELETING INSTRUMENTS -------------
    def instrument_add(self, child, idx):
        AppLogger.print(f"Adding Instrument {idx.name()}")
        id = idx.name().split(' ')[0]  # Isin
        self.instruments[id] = AppBond(id)  # Bond Object

        # Link Instrument Parameters to hedging function:
        for param in ['Notional', 'HedgeInstrument']:
            self.params.param('Instruments').param(id).param(param).sigValueChanged.connect(self.apply_hedge)

        # If a new instrument is added, everything needs to be updated:
        self.params_change_manager()

    def instrument_delete(self, child: InstrumentsGroupParam):
        to_delete = list(set(self.instruments.keys()) - set(child.current_instrument_names()))
        if len(to_delete) > 0:
            del self.instruments[to_delete[0]]
        # If an instrument is deleted, we need to update everything
        self.params_change_manager()

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

    # --------------------- UTILITIES ----------------------------
    def get_selected_date(self):
        date_param = self.params.param('Data Params').param("Date").value()
        date = dt.date(date_param.year(), date_param.month(), date_param.day())
        return date

    def update_candles(self):
        AppLogger.print(f"Calling - update_candles")
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

                if ema_w >= 1:
                    df = df.set_index('Date').ewm(span=ema_w).mean().reset_index()
                new_data.append(df)

            bond.set_current_yield_data(new_data[0])
            bond.set_current_price_data(new_data[1])

        # Once candles have been updated, i need to update all data and plots:
        self.params_change_manager()

    # -------------------------- PLOTS ---------------------------
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
            if not isin_params['Enabled']:
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
            y = yld_data['Close'].values * 100

            x2 = prc_data['Date'].values.astype(np.int64) // 10 ** 9
            y2 = prc_data['Close'].values

            self.yield_plot.plot(x, y, pen=pen)
            self.price_plot.plot(x2, y2, pen=pen)

    def update_volatility(self):
        volatilities = {}
        vola_type = self.params.param('Data Params').param('Volatility Type').value()
        for i, isin in enumerate(self.instruments.keys()):
            app_bond = self.instruments[isin]
            volatilities[isin] = app_bond.compute_volatility(vola_type)
        self.volatilities = np.array([volatilities[isin] for isin in self.isins])
        self.update_maturity_plot()

    def update_maturity_plot(self):
        self.yield_curve_plot.clear()
        self.yield_curve_plot.plot(self.maturities, self.yields * 100, symbol='o', pen=pg.mkPen(color=(206, 241, 229)))
        try:
            popt, pcov = curve_fit(nelson_siegel, self.maturities, self.yields)
            x = np.linspace(np.min(self.maturities), np.max(self.maturities), 100)
            y = nelson_siegel(x, popt[0], popt[1], popt[2], popt[3])
            self.yield_curve_plot.plot(x, y * 100, pen=pg.mkPen(color=(255, 0, 0)))
        except:
            pass

        self.volatility_plot.clear()
        self.volatility_plot.plot(self.maturities, self.volatilities, symbol='o', pen=pg.mkPen(color=(206, 241, 229)))
        self.update_pca_plot()

    def update_pca_plot(self):
        self.pca_plot.clear()
        if 'Components' in self.pca_data.keys():
            n_comps = self.pca_data['Components'].shape[0]
            pens = [pg.mkPen(color=random_color()) for _ in range(n_comps)]
            for i in range(n_comps):
                self.pca_plot.plot(self.pca_data['X'], self.pca_data['Components'][i], pen=pens[i], name=f"Comp {i}")

    def update_pca_computation(self):
        sorted_df = self.yield_series
        n_comps = self.params.param('Data Params').param('PCA Components').value()
        pca = PCA(n_components=n_comps)
        try:
            pca.fit(sorted_df)
            self.pca_data = {'X': self.maturities,
                             'Components': pca.components_}
        except ValueError:
            self.pca_data = {}
        self.update_pca_plot()

    def update_all_instruments_single_property(self, property_name, property_item, to_array=True):
        prop_list = []
        for i, isin in enumerate(self.instruments.keys()):
            isin_params = self.params.param('Instruments').param(isin)
            prop_list.append(isin_params[property_name])
        prop_list = [prop_list[i] for i in self.sorted_indexes]
        if to_array:
            prop_list = np.array(prop_list)
        self.__setattr__(property_item, prop_list)

    def update_all_instrument_properties(self, date=None):
        AppLogger.print("Calling - update_all_instrument_properties")
        maturities = []
        dv01s = []
        durations = []
        convexities = []
        isins = []
        notionals = []
        hedge_notionals = []
        prices = []
        yields = []
        is_hedge = []
        is_enabled = []
        yld_series = []
        price_series = []
        volatilities = []
        corr_price = []
        corr_yield = []
        update_date = self.get_selected_date() if date is None else date
        vola_type = self.params.param('Data Params').param('Volatility Type').value()
        for i, isin in enumerate(self.instruments.keys()):
            app_bond = self.instruments[isin]
            assert isinstance(app_bond, AppBond)
            try:
                isin_params = self.params.param('Instruments').param(isin)
            except KeyError:
                continue

            is_enabled.append(isin_params['Enabled'])
            app_bond.update_data(date=update_date)
            prices.append(app_bond.price)
            yields.append(app_bond.yld)
            maturities.append(app_bond.years_to_maturity)
            dv01s.append(app_bond.dv01)
            durations.append(app_bond.duration)
            convexities.append(app_bond.convexity)
            isins.append(app_bond.instrument_id)
            notionals.append(isin_params['Notional'])
            hedge_notionals.append(self.current_hedge.get(isin, 0))
            is_hedge.append(isin_params['HedgeInstrument'])

            # This is for PCA, contains all Historic Close Yield Data
            yld_data = app_bond.get_current_yield_data()
            prc_data = app_bond.get_current_price_data()
            close_yld = yld_data.set_index('Date')[['Close']]
            close_yld.columns = [app_bond.instrument_id]
            close_prc = prc_data.set_index('Date')[['Close']]
            close_prc.columns = [app_bond.instrument_id]
            yld_series.append(close_yld)
            price_series.append(close_prc)

            # This is for volatility
            volatilities.append(app_bond.compute_volatility(vola_type))

        try:
            yld_series = pd.concat(yld_series, axis=1)
            price_series = pd.concat(price_series, axis=1)
        except ValueError:
            yld_series = None
            price_series = None

        sorted_indexes = np.argsort(maturities)
        self.sorted_indexes = sorted_indexes
        self.isins = [isins[x] for x in sorted_indexes]
        self.is_hedge = [is_hedge[x] for x in sorted_indexes]
        self.is_enabled = np.array([is_enabled[x] for x in sorted_indexes])
        self.maturities = np.array([maturities[x] for x in sorted_indexes])
        self.volatilities = np.array([volatilities[x] for x in sorted_indexes])
        self.prices = np.array([prices[x] for x in sorted_indexes])
        self.yields = np.array([yields[x] for x in sorted_indexes])
        self.notionals = np.array([notionals[x] for x in sorted_indexes])
        self.hedge_notionals = np.array([hedge_notionals[x] for x in sorted_indexes])
        self.dv01s = np.array([dv01s[x] for x in sorted_indexes])
        self.durations = np.array([durations[x] for x in sorted_indexes])
        self.convexities = np.array([convexities[x] for x in sorted_indexes])
        self.yield_series = yld_series[self.isins].fillna(method='ffill') if yld_series is not None else None
        self.price_series = price_series[self.isins].fillna(method='ffill') if price_series is not None else None

    def update_hedge_tab(self):
        # self.apply_hedge()

        multiply_by_notionals = self.params.param("Hedge Options").param('Notional Multiply').value()
        dv01s = self.dv01s
        durations = self.durations
        convexities = self.convexities
        isins = self.isins
        notionals = self.notionals
        hedge_notionals = self.hedge_notionals
        prices = self.prices
        is_hedge = self.is_hedge
        total_notional = notionals + hedge_notionals

        instrument_values = total_notional * prices * 100
        net_value = np.sum(np.abs(total_notional) * prices * 100)

        if net_value == 0:
            net_value = 1E35
            weights = instrument_values
        else:
            weights = instrument_values / net_value


        # -----------------  Dollar Duration  -------------------
        main_pf_dv01 = dv01s * instrument_values * np.invert(is_hedge) * 0.0001
        hedge_pf_dv01 = dv01s * instrument_values * is_hedge * 0.0001
        total_pf_dv01 = dv01s * instrument_values * 0.0001

        # -----------------  Duration  -------------------
        main_pf_duration =  durations * weights * np.invert(is_hedge)
        hedge_pf_duration = durations * weights * is_hedge
        total_pf_duration = durations * weights

        # -----------------  Dollar Convexity  -------------------
        main_pf_convexityd = convexities * weights * prices * np.invert(is_hedge)
        hedge_pf_convexityd = convexities * weights * prices * is_hedge
        total_pf_convexityd = convexities * weights * prices

        # -----------------  Convexity  -------------------
        main_pf_convexity = convexities  * weights * np.invert(is_hedge)
        hedge_pf_convexity = convexities * weights * is_hedge
        total_pf_convexity = convexities * weights

        # Make plots:
        for plot in [self.dv01_plot, self.duration_plot, self.convexity_plot, self.hedge_daily_pl_plot]:
            plot.clear()

        if multiply_by_notionals:
            bargraph_dv01_1 = pg.BarGraphItem(x=self.maturities, height=main_pf_dv01, width=0.2, brush='#43bce8')
            bargraph_dv01_2 = pg.BarGraphItem(x=self.maturities, height=hedge_pf_dv01, width=0.2, brush='#d6564d')
            self.dv01_plot.addItem(bargraph_dv01_1)
            self.dv01_plot.addItem(bargraph_dv01_2)

            bargraph_dur_1 = pg.BarGraphItem(x=self.maturities, height=main_pf_duration, width=0.2, brush='#43bce8')
            bargraph_dur_2 = pg.BarGraphItem(x=self.maturities, height=hedge_pf_duration, width=0.2, brush='#d6564d')
            self.duration_plot.addItem(bargraph_dur_1)
            self.duration_plot.addItem(bargraph_dur_2)

            bargraph_conv_1 = pg.BarGraphItem(x=self.maturities, height=main_pf_convexityd, width=0.2, brush='#43bce8')
            bargraph_conv_2 = pg.BarGraphItem(x=self.maturities, height=hedge_pf_convexityd, width=0.2, brush='#d6564d')
            self.convexity_plot.addItem(bargraph_conv_1)
            self.convexity_plot.addItem(bargraph_conv_2)

        else:
            bargraph1 = pg.BarGraphItem(x=self.maturities, height=dv01s, width=0.2, brush='#43bce8')
            bargraph2 = pg.BarGraphItem(x=self.maturities, height=durations, width=0.2, brush='#43bce8')
            bargraph3 = pg.BarGraphItem(x=self.maturities, height=convexities, width=0.2, brush='#43bce8')
            self.dv01_plot.addItem(bargraph1)
            self.duration_plot.addItem(bargraph2)
            self.convexity_plot.addItem(bargraph3)


        # Compute P&L Variations
        self.build_pnl_variations_plot()

        # Plot
        self.dv01_plot.setTitle(f"<font size='5'> DV01 : {round(np.sum(total_pf_dv01), 2)}</font>")
        self.duration_plot.setTitle(f"<font size='5'> Duration : {str(round(np.sum(total_pf_duration), 2))} </font>")
        self.convexity_plot.setTitle(f"<font size='5'> Convexity : {str(round(np.sum(total_pf_convexityd), 2))}</font>")

        # Build The Table:
        data = []
        for i in range(len(isins)):
            data.append((isins[i],
                         dv01s[i],
                         total_pf_dv01[i],
                         durations[i],
                         total_pf_duration[i],
                         convexities[i],
                         total_pf_convexity[i],
                         notionals[i],
                         hedge_notionals[i],
                         is_hedge[i])
                        )

        data_array = np.array(data,
                              dtype=[('Isin', object),
                                     ('DV01', float),
                                     ('DV01 PF', float),
                                     ('Duration', float),
                                     ('Duration PF', float),
                                     ('Convexity', float),
                                     ('Convexity PF', float),
                                     ('Notional', float),
                                     ('HedgeNotional', float),
                                     ('IsHedge', bool)])

        self.hedge_data_table.setData(data_array)

        print("Current Portfolio Variance:", self.compute_portfolio_variance())

    def build_pnl_variations_plot(self):
        # X Axis
        dates = self.price_series.index[1:]
        x = pd.Series(dates).view(np.int64) // 10 ** 9

        # Y Axis
        movements = self.price_series.diff().dropna().values
        portfolio_pl_changes = np.sum((self.notionals) * 100 * movements, axis=1)
        hedged_pl_changes = np.sum((self.notionals+self.hedge_notionals)*100*movements, axis=1)

        # Plot
        self.hedge_daily_pl_plot.plot(x, portfolio_pl_changes, pen=pg.mkPen(color='#43bce8'), name="Original")
        self.hedge_daily_pl_plot.plot(x, hedged_pl_changes, pen=pg.mkPen(color='#d6564d'), name="Hedged")

    def update_correlation_plots(self):
        self.yield_corr_plot.clear()
        if len(self.instruments) < 2:
            return
        self.plot_correlogram(self.yield_corr_plot, self.yield_series.corr())
        self.plot_correlogram(self.price_corr_plot,  self.price_series.corr())

    def plot_correlogram(self, parent, data):
        correlogram = pg.ImageItem()
        tr = QtGui.QTransform().translate(-0.5, -0.5)
        correlogram.setTransform(tr)
        correlogram.setImage(data.values)
        parent.invertY(True)  # orient y axis to run top-to-bottom
        parent.setDefaultPadding(0.0)  # plot without padding data range
        parent.addItem(correlogram)  # display correlogram
        parent.showAxes(True, showValues=(True, True, False, False), size=20)
        ticks = [(idx, str(idx)) for idx, label in enumerate(data.columns)]
        for side in ('left', 'top', 'right', 'bottom'):
            parent.getAxis(side).setTicks((ticks, []))  # add list of major ticks; no minor ticks
        parent.getAxis('bottom').setHeight(10)  # include some additional space at bottom of figure
        colorMap = pg.colormap.get("inferno", source='matplotlib')  # choose perceptually uniform, diverging color map
        bar = pg.ColorBarItem(values=(-1, 1), colorMap=colorMap)
        bar.setImageItem(correlogram, insert_in=parent)

    # ---------------------------- HEDGE ----------------------------------
    def apply_hedge(self):
        notionals_change = self.instrument_properties_have_changed('latest_notionals', 'Notional')
        h_instruments_change = self.instrument_properties_have_changed('current_hedge_instruments', 'HedgeInstrument')
        method_change = self.hedge_method_changed()
        date_changed = self.date_picker_changed()
        if notionals_change or h_instruments_change or date_changed or method_change:
            # Need to Update the vectors indicating which bonds are hedge instruments and the notional
            self.update_all_instruments_single_property(property_name='HedgeInstrument', property_item="is_hedge")
            self.update_all_instruments_single_property(property_name='Notional', property_item="notionals")
            self.hedge_funcs[self.params.param('Hedge Options').param('Hedge Type').value()]()
            self.update_hedge_tab()  # Update thed Hedge Tab in the App

    def duration_hedge(self):
        is_hedge = self.is_hedge
        if np.sum(self.prices*self.durations*is_hedge) == 0:
            self.current_hedge = {}
        else:
            try:
                hedge_instrument = [self.isins[i] for i in range(len(self.isins)) if is_hedge[i]==True]
                if len(hedge_instrument) > 1:
                    AppLogger.print(f"More than one hedge instrument selected, defaulting to {hedge_instrument}")
                    is_hedge = [1  if isin == hedge_instrument[0] else 0 for isin in self.isins ]

                hnot = -np.dot(self.prices * self.notionals, self.dv01s) / (np.sum(self.prices * self.dv01s * is_hedge))
                self.current_hedge = {hedge_instrument[0]: hnot}
            except:
                self.current_hedge = {}
        self.hedge_notionals = np.array([self.current_hedge.get(isin, 0) for isin in self.isins])

    def quadratic_hedge(self):
        is_hedge = self.is_hedge
        if np.sum(is_hedge) != 2:
            AppLogger.print("You need exactly 2 instruments to hedge via Duration-COnvexity")
            self.current_hedge = {}

        else:
            hedge_instrument = [self.isins[i] for i in range(len(self.isins)) if is_hedge[i] == True]
            idx = [i for i in range(len(self.isins)) if is_hedge[i] == True]
            is_hedge = [1 if isin in hedge_instrument else 0 for isin in self.isins]

            a = np.array([
                [self.prices[idx[0]] * self.durations[idx[0]], self.prices[idx[1]] * self.durations[idx[1]]],
                [self.prices[idx[0]] * self.durations[idx[0]] * self.maturities[idx[0]], self.prices[idx[1]] * self.convexities[idx[1]] * self.maturities[idx[1]]]
            ])
            b = np.array([-np.dot(self.prices * self.notionals, self.durations),
                          -np.dot(self.prices * self.notionals, self.durations * self.maturities)
                          ])
            x = np.linalg.solve(a, b)

            self.current_hedge = {hedge_instrument[0]: x[0],
                                  hedge_instrument[1]: x[1]}

        self.hedge_notionals = np.array([self.current_hedge.get(isin, 0) for isin in self.isins])

    def cubic_hedge(self):
        is_hedge = self.is_hedge
        if np.sum(is_hedge) != 3:
            AppLogger.print("You need exactly 3 instruments to hedge via Duration-COnvexity")
            self.current_hedge = {}
        else:
            hedge_instrument = [self.isins[i] for i in range(len(self.isins)) if is_hedge[i] == True]
            idx = [i for i in range(len(self.isins)) if is_hedge[i] == True]
            main_portfolio_indexes = [i for i in range(len(self.isins)) if self.notionals[i] != 0]

            T = np.max([self.maturities[i] for i in main_portfolio_indexes])
            NP = np.sum(self.notionals*self.prices)
            D = np.sum(self.notionals*self.durations) / np.sum(self.notionals)
            C = np.sum(self.convexities*self.durations) / np.sum(self.durations)

            PA, DA, TA = self.prices[idx[0]], self.durations[idx[0]], self.maturities[idx[0]]
            PB, DB, TB = self.prices[idx[1]], self.durations[idx[1]], self.maturities[idx[1]]
            PC, DC, TC = self.prices[idx[2]], self.durations[idx[2]], self.maturities[idx[2]]

            NA = - NP*D/(PA*DA) * ((T-TC)*(T-TB))/ ((TB-TA)*(TC-TA))
            NB = - NP*D/(PB*DB) * ((T-TC)*(T-TA))/((TB-TA)*(TB-TC))
            NC = - NP*D/(PC*DC) * ((TB-T)*(T-TA))/((TC-TA)*(TB-TC))

            self.current_hedge = {hedge_instrument[0]: NA,
                                  hedge_instrument[1]: NB,
                                  hedge_instrument[2]: NC
                                  }

        self.hedge_notionals = np.array([self.current_hedge.get(isin, 0) for isin in self.isins])

    def duration_convexity_hedge(self):
        is_hedge = self.is_hedge
        if np.sum(is_hedge) != 2:
            AppLogger.print("You need exactly 2 instruments to hedge via Duration-COnvexity")
            self.current_hedge = {}

        else:
            hedge_instrument = [self.isins[i] for i in range(len(self.isins)) if is_hedge[i]==True]
            idx = [i for i in range(len(self.isins)) if is_hedge[i] == True]
            is_hedge = [1 if isin in hedge_instrument else 0 for isin in self.isins]

            a = np.array([
                [self.prices[idx[0]]*self.dv01s[idx[0]], self.prices[idx[1]]*self.dv01s[idx[1]]],
                [self.prices[idx[0]]*self.convexities[idx[0]]*self.prices[idx[0]], self.prices[idx[1]]*self.convexities[idx[1]]*self.prices[idx[1]]]
            ])
            b = np.array([-np.dot(self.prices*self.notionals, self.dv01s),
                          -np.dot(self.prices*self.notionals, self.prices*self.convexities)
                          ])
            x = np.linalg.solve(a, b)

            self.current_hedge = {hedge_instrument[0]: x[0],
                                  hedge_instrument[1]: x[1]}

        self.hedge_notionals = np.array([self.current_hedge.get(isin, 0) for isin in self.isins])

    def duration_convexity_dv01_hedge(self):
        is_hedge = self.is_hedge
        if np.sum(is_hedge) != 3:
            AppLogger.print("You need exactly 2 instruments to hedge via Duration-COnvexity")
            self.current_hedge = {}

        else:
            hedge_instrument = [self.isins[i] for i in range(len(self.isins)) if is_hedge[i]==True]
            idx = [i for i in range(len(self.isins)) if is_hedge[i] == True]
            is_hedge = [1 if isin in hedge_instrument else 0 for isin in self.isins]

            a = np.array([
                [self.prices[i] * self.durations[i] for i in idx],
                [self.prices[i] * self.convexities[i] for i in idx],
                [self.prices[i] * self.dv01s[i] for i in idx]
            ])
            b = np.array([-np.dot(self.prices * self.notionals, self.durations),
                          -np.dot(self.prices * self.notionals, self.convexities),
                          -np.dot(self.prices * self.notionals, self.dv01s)
                          ])
            x = np.linalg.solve(a, b)

            self.current_hedge = {hedge_instrument[0]: x[0],
                                  hedge_instrument[1]: x[1],
                                  hedge_instrument[2]: x[2]}

        self.hedge_notionals = np.array([self.current_hedge.get(isin, 0) for isin in self.isins])

    def variance_minimization_hedge(self):
        hedge_indexes = np.where(self.is_hedge==1)[0]
        hedge_instrument = [self.isins[i] for i in range(len(self.isins)) if self.is_hedge[i] == True]

        def variance_minimization_func(args):
            notionals = self.notionals.copy()
            for i, idx in enumerate(hedge_indexes):
                notionals[idx] = args[i]
            pf_var = self.compute_portfolio_variance(notionals)
            return pf_var

        minimization = minimize(variance_minimization_func, x0=[0 for _ in hedge_indexes], tol=1E-8, method='CG')
        print(minimization.message)
        self.current_hedge = {hedge_instrument[i]: minimization.x[i] for i in range(len(hedge_instrument))}
        self.hedge_notionals = np.array([self.current_hedge.get(isin, 0) for isin in self.isins])

    def compute_portfolio_variance(self, notionals=None):
        if notionals is None:
            notionals = self.notionals + self.hedge_notionals

        vola = np.diag(self.volatilities)
        corr = self.price_series.corr().values
        covariance = np.matmul(np.matmul(vola, corr), vola)

        instrument_values = notionals * self.prices * 100
        net_value = np.sum(np.abs(notionals) * self.prices * 100)

        if net_value != 0:
            weights = instrument_values / net_value
        else:
            weights = instrument_values

        portfolio_variance = np.dot(np.dot(weights, covariance), weights)
        return portfolio_variance

    # ------------------------- CHANGE SIGNALS -----------------------------
    def instrument_properties_have_changed(self, property, instrument_param):
        attr = {}
        for i, isin in enumerate(self.instruments.keys()):
            app_bond = self.instruments[isin]
            assert isinstance(app_bond, AppBond)
            try:
                isin_params = self.params.param('Instruments').param(isin)
            except KeyError:
                continue

            attr[app_bond.instrument_id] = isin_params[instrument_param]
        if attr != self.__getattribute__(property):
            self.__setattr__(property, attr)
            return True
        else:
            return False

    def date_picker_changed(self):
        if self.selected_date == self.get_selected_date():
            return False
        else:
            self.selected_date = self.get_selected_date()
            return True

    def hedge_method_changed(self):
        method = self.params.param('Hedge Options').param('Hedge Type').value()
        if self.current_hedge_method == method:
            return False
        else:
            self.current_hedge_method = method
            return True

    # ------------------------- PARAM CHANGE -------------------------------
    def params_change_manager(self):
        AppLogger.print("Calling - params_change_manager")

        if len(self.instruments) == 0:
            return

        # Update Data:
        self.update_all_instrument_properties()
        self.update_pca_computation()

        # Tab 1 Update:
        self.update_price_plot()
        self.update_maturity_plot()
        self.update_correlation_plots()
        self.update_yield_surface_plot()

        # Tab 2 Update:
        self.update_hedge_tab()


if __name__ == "__main__":
    from pyqtgraph.examples.relativity import relativity

    AppLogger.verbosity = 0

    test = False
    if test:
        import pyqtgraph.examples

        pyqtgraph.examples.run()

    else:
        app = QApplication(sys.argv)
        win = Window()
        win.show()
        sys.exit(app.exec())
