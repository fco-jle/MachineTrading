import QuantLib as ql
import prettytable as pt
import datetime as dt
import numpy as np


class FixedRateBond:
    def __init__(self, issue_date, maturity_date, coupon, settlement_days=2,
                 calendar=None, payment_convention=None, end_of_month=False,
                 coupon_period=ql.Semiannual, compounding=ql.Compounded):

        self.settlementDays = settlement_days
        face_amount = 100.0
        self.issue_date = self._date_to_quantlib(issue_date)
        self.maturity_date = self._date_to_quantlib(maturity_date)
        self.frequency = coupon_period
        self.compounding = compounding
        tenor = ql.Period(self.frequency)

        self.calendar = calendar if calendar is not None else ql.NullCalendar()
        convention = ql.Unadjusted
        maturity_date_convention = convention
        rule = ql.DateGeneration.Backward
        coupons = ql.DoubleVector(1)
        coupons[0] = coupon / 100.0
        payment_convention = payment_convention if payment_convention else ql.Unadjusted

        schedule = ql.Schedule(self.issue_date, self.maturity_date, tenor, self.calendar,
                               convention, maturity_date_convention, rule, end_of_month)
        self.accrualDayCounter = ql.ActualActual(ql.ActualActual.Bond, schedule)
        self.pricingDayCounter = ql.ActualActual(ql.ActualActual.Bond)
        self.bond = ql.FixedRateBond(self.settlementDays, face_amount,
                                     schedule, coupons, self.accrualDayCounter,
                                     payment_convention)

    @staticmethod
    def _date_to_quantlib(date):
        if isinstance(date, dt.date):
            return ql.Date(date.day, date.month, date.year)
        elif isinstance(date, ql.Date):
            return date
        else:
            raise ValueError

    def set_evaluation_date(self, evaluation_date):
        d = self._date_to_quantlib(evaluation_date)
        ql.Settings.instance().evaluationDate = d

    def cash_flows(self):
        for c in self.bond.cashflows():
            print('%20s %12f' % (c.date(), c.amount()))

    def _set_pricing_engine(self, bond_yield, day_counter=None, compounding_period=None, settlement_days=None):
        compounding_period = compounding_period if compounding_period is not None else ql.Annual
        accrual_day_counter = day_counter if day_counter is not None else self.accrualDayCounter
        settlement_days = settlement_days if settlement_days is not None else self.settlementDays
        flat_curve = ql.FlatForward(settlement_days,
                                    self.calendar,
                                    bond_yield,
                                    accrual_day_counter,
                                    self.compounding,
                                    compounding_period)
        term_structure = ql.YieldTermStructureHandle(flat_curve)
        term_structure.enableExtrapolation()
        engine = ql.DiscountingBondEngine(term_structure)
        self.bond.setPricingEngine(engine)

    def bond_yield(self, price, eval_date=None):
        if eval_date:
            self.set_evaluation_date(eval_date)
        yld = self.bond.bondYield(price, self.pricingDayCounter, self.compounding, self.frequency)
        return yld

    def clean_price(self, bond_yield, eval_date=None, errors='raise'):
        if eval_date:
            self.set_evaluation_date(eval_date)
        try:
            prc = self.bond.cleanPrice(bond_yield, self.pricingDayCounter, self.compounding, ql.Annual)
        except RuntimeError:
            if errors == 'coerce':
                prc = np.nan
            elif errors == 'raise':
                raise RuntimeError
            else:
                raise ValueError("Invalid errors value")
        return prc

    def dirty_price(self, bond_yield, eval_date=None):
        if eval_date:
            self.set_evaluation_date(eval_date)
        prc = self.bond.dirtyPrice(bond_yield, self.pricingDayCounter, self.compounding, self.frequency)
        return prc

    def npv(self, bond_yield, eval_date=None):
        if eval_date:
            self.set_evaluation_date(eval_date)
        self._set_pricing_engine(bond_yield, day_counter=self.pricingDayCounter, compounding_period=self.frequency)
        npv = self.bond.NPV()
        return npv

    def dv01_from_yield(self, bond_yield, eval_date=None):
        p1 = self.clean_price(bond_yield=bond_yield, eval_date=eval_date)
        p2 = self.clean_price(bond_yield=bond_yield + 0.01 / 100, eval_date=eval_date)
        dv01 = p1 - p2
        return dv01

    def dv01_from_price(self, price, eval_date=None):
        yld = self.bond_yield(price, eval_date)
        dv01 = price - self.clean_price(yld + 0.01/100, eval_date)
        return dv01

    def accrued_amount(self, eval_date=None):
        ql_eval_date = self._date_to_quantlib(eval_date)
        return self.bond.accruedAmount(ql_eval_date)

    def duration(self, bond_yield, kind=ql.Duration.Simple):
        return ql.BondFunctions.duration(
            self.bond,
            bond_yield,
            self.pricingDayCounter,
            self.compounding,
            self.frequency,
            kind)

    def duration_simple(self, bond_yield):
        kind = ql.Duration.Simple
        return self.duration(bond_yield, kind)

    def duration_modified(self, bond_yield):
        kind = ql.Duration.Modified
        return self.duration(bond_yield, kind)

    def duration_macaulay(self, bond_yield):
        kind = ql.Duration.Macaulay
        return self.duration(bond_yield, kind)

    def convexity(self, bond_yield):
        return ql.BondFunctions.convexity(
            self.bond,
            bond_yield,
            self.pricingDayCounter,
            self.compounding,
            self.frequency)

    def bpv(self, price, eval_date=None):
        if eval_date:
            self.set_evaluation_date(eval_date)

        bond_yield = self.bond_yield(price, eval_date)
        return ql.BondFunctions.basisPointValue(
            self.bond,
            bond_yield,
            self.pricingDayCounter,
            self.compounding,
            self.frequency)

    def years_to_maturity(self, eval_date=None):
        eval_date = self._date_to_quantlib(dt.date.today()) if eval_date is None else self._date_to_quantlib(eval_date)
        ytm = (self.maturity_date - eval_date) / 365.25
        return ytm

    def futures_contract_conversion_factor(self, bond_price, futures_price, futures_delivery_date, futures_coupon=0.06):
        """
        The Conversion Factor for a cash Treasury security is the price of that security that would make
        its yield to the futures delivery date equal to its coupon rate.
        """
        cf = self.clean_price(bond_yield=futures_coupon, eval_date=futures_delivery_date) / 100
        adjusted_futures_price = futures_price * cf
        basis = bond_price - adjusted_futures_price
        return cf, basis


class BTP(FixedRateBond):
    def __init__(self, issue_date, maturity_date, coupon):
        super(BTP, self).__init__(
            issue_date=issue_date,
            maturity_date=maturity_date,
            coupon=coupon,
            settlement_days=2,
            calendar=ql.NullCalendar(),
            payment_convention=ql.ModifiedFollowing,
            coupon_period=ql.Semiannual,
            end_of_month=True
        )


class FixedRateBondEurex(FixedRateBond):
    def __init__(self, issue_date, maturity_date, coupon=0.06):
        super(FixedRateBondEurex, self).__init__(
            issue_date=issue_date,
            maturity_date=maturity_date,
            coupon=coupon,
            settlement_days=0,
            calendar=ql.Germany(),
            payment_convention=ql.ModifiedFollowing,
            end_of_month=False
        )

    def clean_price(self, bond_yield, eval_date=None):
        if eval_date:
            self.set_evaluation_date(eval_date)
        prc = self.bond.cleanPrice(bond_yield, self.pricingDayCounter, ql.Compounded, ql.Annual)
        return prc


if __name__ == '__main__':
    # TEST = 'ComputePrice'
    # TEST = 'ForwardYield'
    # TEST = 'CTD'
    # TEST = "Borsa"
    TEST = "BPV"
    # TEST = "DV01"

    if TEST == "BPV":
        coupon, maturity, issue = (2.50, ql.Date(1, 12, 2024), ql.Date(1, 9, 2014))
        p = 103.573
        dv01 = 2.546
        s2 = BTP(issue, maturity, coupon)
        s2.cash_flows()  # 0.623288
        yld = s2.bond_yield(price=p, eval_date=ql.Date(12, 5, 2022)) * 100  #
        modified_duration = s2.duration_modified(yld / 100)
        simple_duration = s2.duration_simple(yld / 100)
        bpv = -s2.bpv(p, eval_date=ql.Date(12, 5, 2022)) * 100

    if TEST == "DV01":
        coupon, maturity, issue = (0.90, ql.Date(1, 4, 2031), ql.Date(1, 10, 2020))
        p = 83.55
        y = 3.04
        s2 = BTP(issue, maturity, coupon, )
        test = s2.bond_yield(price=p, eval_date=ql.Date(6, 5, 2022)) * 100  # 3.04
        modified_duration = s2.duration_modified(y / 100)
        simple_duration = s2.duration_simple(y/100)
        bpv = -s2.bpv(p, eval_date=ql.Date(6, 5, 2022))*100
        dv01_2 = s2.dv01_from_price(p, eval_date=ql.Date(6, 5, 2022))*100
        dv01_from_duration = p*modified_duration/100
        dv01_from_duration_simple = p * simple_duration / 100

        p1 = s2.clean_price(3.04/100)
        p2 = s2.clean_price(3.04 / 100 + 0.1/100)
        p0 = s2.clean_price(3.04 / 100 - 0.1 / 100)

        diff1 = (p1 - p2) / 0.1
        diff2 = (p0 - p1) / 0.1
        diff = (diff1+diff2)/2


    if TEST == "Borsa":
        coupon, maturity, issue = (0.90, ql.Date(1, 4, 2031), ql.Date(1, 10, 2020))
        p = 83.55
        y = 3.04
        s2 = BTP(issue, maturity, coupon, )
        test = s2.bond_yield(price=p, eval_date=ql.Date(6, 5, 2022))*100  # 3.04
        test2 = s2.clean_price(bond_yield=3.037/100, eval_date=ql.Date(6, 5, 2022))  # 83.55
        test3 = s2.dirty_price(bond_yield=3.037/100, eval_date=ql.Date(6, 5, 2022))
        test4 = s2.accrued_amount(eval_date=ql.Date(6, 5, 2022))
        test5 = s2.npv(bond_yield=test/100, eval_date=ql.Date(6, 5, 2022))

    if TEST == 'ForwardYield':
        """
        The Yield for a futures contract is calculated as the yield to maturity of a cash security with the following 
        specifications:
        - Settlement Date = last delivery day for the futures contract
        - Maturity Date = maturity date of the CTD cash security
        - Coupon Rate = coupon rate per annum of the CTD cash security
        - Bond Price = (futures price * conversion factor for CTD cash security) + (accrued Coupon interest on CTD 
                                                                                    cash security, from latest Coupon 
                                                                                    payment date to Settlement Date)
        - Coupon Frequency = (Coupon Rate / 2) paid semiannually
        - Day Count Basis = (actual / actual)
        - Par Value = 100
        """

        coupon, maturity, issue, price = (0.90, ql.Date(1, 4, 2031), ql.Date(1, 10, 2020), 90.81)
        f_price = 137.54
        f_delivery = ql.Date(10, 6, 2022)
        settle = f_delivery

        s = FixedRateBondEurex(issue, maturity, coupon)
        cf, basis = s.futures_contract_conversion_factor(price, f_price, f_delivery, 0.06)
        expected_bond_price = (f_price * cf) + s.accrued_amount(f_delivery)
        fwd_yld = s.bond_yield(price=expected_bond_price, eval_date=settle)
        print("%-15s = %lf" % ("Forward Price", f_price))
        print("%-15s = %lf" % ("Forward Yield", fwd_yld*100))

    if TEST == 'ComputePrice':
        coupon, maturity, issue = (0.90, ql.Date(1, 4, 2031), ql.Date(1, 10, 2020))
        s2 = BTP(issue, maturity, coupon)
        test = s2.bond_yield(price=91.04, eval_date=ql.Date(1, 4, 2022))  # 2.0
        test2 = s2.clean_price(bond_yield=test, eval_date=ql.Date(1, 4, 2022))  # 91.04
        test4 = s2.dirty_price(bond_yield=test, eval_date=ql.Date(1, 4, 2022))  # 91.04
        test3 = s2.npv(bond_yield=test, eval_date=ql.Date(1, 4, 2022))

        print("%-15s = %1.3f" % ("Expected Yield", 2.0))
        print("%-15s = %1.3f" % ("Bond Yield", round(test * 100, 2)))
        print("%-15s = %1.3f" % ("Expected Price", 91.04))
        print("%-15s = %1.3f" % ("Clean Price", test2))
        print("%-15s = %1.3f" % ("Dirty Price", test4))
        print("%-15s = %1.3f" % ("NPV", test3))
        print("")

        p = 88.01
        y = 0.0203
        coupon, maturity, issue = (0.60, ql.Date(1, 8, 2031), ql.Date(23, 2, 2021))
        s2 = BTP(issue, maturity, coupon)
        ttest = s2.bond_yield(price=p, eval_date=ql.Date(1, 4, 2022))  # 2.0
        ttest2 = s2.clean_price(bond_yield=ttest, eval_date=ql.Date(1, 4, 2022))  # 88.01
        ttest4 = s2.dirty_price(bond_yield=ttest, eval_date=ql.Date(1, 4, 2022))  # 88.41
        ttest3 = s2.npv(bond_yield=ttest, eval_date=ql.Date(1, 4, 2022))

        print("%-15s = %1.3f" % ("Expected Yield", y * 100))
        print("%-15s = %1.3f" % ("Bond Yield", round(ttest * 100, 2)))
        print("%-15s = %1.3f" % ("Expected Price", p))
        print("%-15s = %1.3f" % ("Clean Price", ttest2))
        print("%-15s = %1.3f" % ("Dirty Price", ttest4))
        print("%-15s = %1.3f" % ("NPV", ttest3))
        print("")

    if TEST == 'CTD':
        """
        The holder of a short position in a Treasury futures contract must deliver a cash Treasury security to the 
        holder of the offsetting long futures position upon contract expiration.  There are typically several cash 
        securities available that fulfill the specification of the futures contract.  Because of accrued interest, 
        differing maturities, etc. of the various cash securities, there are differing cash flows associated with 
        the deliver process.  The cash security with the lowest cash flow cost is known as the Cheapest to Deliver.
        """

        # A día 2022-04-01
        date = ql.Date(1, 4, 2022)
        basket = [(0.90, ql.Date(1, 4, 2031), ql.Date(1, 10, 2020), 90.81),  # IT0005422891  ->  0.659620
                  (0.60, ql.Date(1, 8, 2031), ql.Date(23, 2, 2021), 88.5),  # IT0005436693  ->  0.628867
                  (0.95, ql.Date(1, 12, 2031), ql.Date(1, 6, 2021), 91.0),  # IT0005449969  ->  0.643888
                  (0.95, ql.Date(1, 6, 2032), ql.Date(1, 11, 2021), 90.09)  # IT0005466013  ->  0.630012
                  ]

        f_price = 137.54
        f_delivery = ql.Date(10, 6, 2022)

        securities = []
        min_basis = 100
        min_basis_index = -1
        for i, b in enumerate(basket):
            coupon, maturity, issue, price = b
            s = FixedRateBondEurex(issue, maturity, coupon)
            cf, basis = s.futures_contract_conversion_factor(price, f_price, f_delivery)
            if basis < min_basis:
                min_basis = basis
                min_basis_index = i
            securities.append((s, cf))

        ctd_info = basket[min_basis_index]
        ctd_bond, ctd_cf = securities[min_basis_index]
        ctd_price = ctd_info[3]

        print("%-30s = %lf" % ("Minimum Basis", min_basis))
        print("%-30s = %lf" % ("Conversion Factor", ctd_cf))
        print("%-30s = %lf" % ("Expected Conversion Factor", 0.659620))
        print("%-30s = %lf" % ("Coupon", ctd_info[0]))
        print("%-30s = %s" % ("Maturity", ctd_info[1]))
        print("%-30s = %lf" % ("Price", ctd_info[3]))
