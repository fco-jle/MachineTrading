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
from MachineTrading.TSA.volatility import garman_klass, standard_dev, close_to_close, parkinson, garman_klass_yang_zhang


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

    def update_data(self, date):
        if date == self.current_data_date:
            AppLogger.print(f"Calling - update_data for isin {self.instrument_id} - Data is up to date - Skipping")
            return

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
        self.dv01 = self.bond.dv01(bond_yield=self.yld, bond_price=self.price, eval_date=date)
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
        self.current_hedge = {}
        self.current_hedge_instruments = {}
        self.objectGroup = InstrumentsGroupParam()

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

    # ------------------------ SETUP -----------------------------
    def setup_gui(self):
        self.splitter = QtWidgets.QSplitter()
        self.splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.ui.plotsLayout.addWidget(self.splitter)

        # Configure Object Time Series Plots
        self.time_series_plots = pg.GraphicsLayoutWidget()
        self.maturity_plot = pg.GraphicsLayoutWidget()
        self.correlation_plot = pg.GraphicsLayoutWidget()

        self.splitter.addWidget(self.time_series_plots)
        self.splitter.addWidget(self.correlation_plot)
        self.splitter.addWidget(self.maturity_plot)
        self.splitter.setSizes([int(self.width() * 0.5), int(self.width() * 0.25), int(self.width() * 0.25)])

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
        self.params.param('Data Params').param('Yield Curve Date').sigValueChanged.connect(self.params_change_manager)
        self.params.param('Data Params').param('PCA Components').sigValueChanged.connect(self.update_pca_computation)
        self.params.param('Data Params').param('Volatility Type').sigValueChanged.connect(self.update_volatility)

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
        date_param = self.params.param('Data Params').param("Yield Curve Date").value()
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

        self.volatility_plot.clear()
        self.volatility_plot.plot(self.maturities, self.volatilities, symbol='o', pen=pg.mkPen(color=(206, 241, 229)))

        self.update_pca_plot()

    def update_pca_plot(self):
        self.pca_plot.clear()
        if 'Components' in self.pca_data.keys():
            n_comps = self.pca_data['Components'].shape[0]
            pens = [pg.mkPen(color=random_color()) for _ in range(n_comps)]
            for i in range(n_comps):
                self.pca_plot.plot(self.pca_data['X'], self.pca_data['Components'][i], pen=pens[i])

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

    def update_all_instrument_properties(self):
        AppLogger.print("Calling - update_all_instrument_properties")
        maturities = []
        dv01s = []
        durations = []
        convexities = []
        isins = []
        notionals = []
        prices = []
        yields = []
        is_hedge = []
        is_enabled = []
        yld_series = []
        price_series = []
        volatilities = []
        corr_price = []
        corr_yield = []
        vola_type = self.params.param('Data Params').param('Volatility Type').value()
        for i, isin in enumerate(self.instruments.keys()):
            app_bond = self.instruments[isin]
            assert isinstance(app_bond, AppBond)
            try:
                isin_params = self.params.param('Instruments').param(isin)
            except KeyError:
                continue

            is_enabled.append(isin_params['Enabled'])
            app_bond.update_data(date=self.get_selected_date())
            prices.append(app_bond.price)
            yields.append(app_bond.yld)
            maturities.append(app_bond.years_to_maturity)
            dv01s.append(app_bond.dv01)
            durations.append(app_bond.duration)
            convexities.append(app_bond.convexity)
            isins.append(app_bond.instrument_id)
            notionals.append(isin_params['Notional'] + self.current_hedge.get(isin, 0))
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
        self.isins = [isins[x] for x in sorted_indexes]
        self.is_hedge = [is_hedge[x] for x in sorted_indexes]
        self.is_enabled = np.array([is_enabled[x] for x in sorted_indexes])
        self.maturities = np.array([maturities[x] for x in sorted_indexes])
        self.volatilities = np.array([volatilities[x] for x in sorted_indexes])
        self.prices = np.array([prices[x] for x in sorted_indexes])
        self.yields = np.array([yields[x] for x in sorted_indexes])
        self.notionals = np.array([notionals[x] for x in sorted_indexes])
        self.dv01s = np.array([dv01s[x] for x in sorted_indexes])
        self.durations = np.array([durations[x] for x in sorted_indexes])
        self.convexities = np.array([convexities[x] for x in sorted_indexes])
        self.yield_series = yld_series[self.isins].fillna(method='ffill') if yld_series is not None else None
        self.price_series = price_series[self.isins].fillna(method='ffill') if price_series is not None else None

    def update_hedge_plot(self):
        self.apply_hedge()
        multiply_by_notionals = self.params.param("Hedge Options").param('Notional Multiply').value()

        dv01s = self.dv01s
        durations = self.durations
        convexities = self.convexities
        isins = self.isins
        notionals = self.notionals
        prices = self.prices
        is_hedge = self.is_hedge

        # Check If we want raw metrics or portfolio metrics
        portfolio_dv01 = dv01s * notionals / 100
        portfolio_duration = 0.0
        portfolio_convexity = 0.0
        if np.sum(notionals)!=0:
            portfolio_duration = (durations * notionals * prices) / (100 * np.sum(notionals))
            portfolio_convexity = (convexities * notionals * prices) / (100 * np.sum(notionals))

        # Make plots:
        for plot in [self.dv01_plot, self.duration_plot, self.convexity_plot]:
            plot.clear()

        bargraph1 = pg.BarGraphItem(x=self.maturities, height=dv01s, width=0.3, brush='#43bce8')
        bargraph2 = pg.BarGraphItem(x=self.maturities, height=durations, width=0.3, brush='#43bce8')
        bargraph3 = pg.BarGraphItem(x=self.maturities, height=convexities, width=0.3, brush='#43bce8')

        self.dv01_plot.addItem(bargraph1)
        self.duration_plot.addItem(bargraph2)
        self.convexity_plot.addItem(bargraph3)

        # Build The Table:
        data = []
        for i in range(len(isins)):
            data.append((isins[i],
                         dv01s[i],
                         durations[i],
                         convexities[i],
                         notionals[i],
                         is_hedge[i])
                        )

        data_array = np.array(data,
                              dtype=[('Isin', object),
                                     ('DV01', float),
                                     ('Duration', float),
                                     ('Convexity', float),
                                     ('Notional', float),
                                     ('IsHedge', bool)])
        self.hedge_data_table.setData(data_array)

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
        date_changed = self.date_picker_changed()
        if notionals_change or h_instruments_change or date_changed:
            self.dv01_hedge()

    def dv01_hedge(self):
        hedge_instrument = ""
        NP = 0
        D = 0
        PA = 0
        DA = 0
        notionals = []
        for i, isin in enumerate(self.instruments.keys()):
            app_bond = self.instruments[isin]
            assert isinstance(app_bond, AppBond)
            try:
                isin_params = self.params.param('Instruments').param(isin)
            except KeyError:
                continue

            # date = self.get_selected_date()
            # app_bond.update_data(date=date)

            if isin_params['HedgeInstrument']:
                if PA == 0:
                    PA = app_bond.price
                    DA = app_bond.duration
                    hedge_instrument = app_bond.instrument_id
                else:
                    print("More than one hedge instrument selected for Duration hedge, ignoring")
            else:
                NP += isin_params['Notional'] * app_bond.price / 100
                D += isin_params['Notional'] * app_bond.duration
                notionals.append(isin_params['Notional'])

        if np.sum(notionals) != 0:
            D /= np.sum(notionals)

        if DA != 0:
            NA = - NP * D / (PA * DA) * 100
        else:
            NA = 0
        self.current_hedge = {hedge_instrument: NA}

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
        latest = self.selected_date
        current = self.get_selected_date()

        if latest == current:
            return False
        else:
            self.selected_date = current
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

        # Tab 2 Update:
        self.update_hedge_plot()


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
