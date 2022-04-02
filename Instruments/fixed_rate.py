import QuantLib as ql
import prettytable as pt
import datetime as dt
import numpy as np


class FixedRateBond:
    def __init__(self, issue_date, maturity_date, coupon, settlement_days=2, calendar=None):
        self.settlementDays = settlement_days
        face_amount = 100.0
        self.issue_date = self._date_to_quantlib(issue_date)
        self.maturity_date = self._date_to_quantlib(maturity_date)
        self.frequency = ql.Semiannual
        self.compounding = ql.Compounded
        tenor = ql.Period(ql.Semiannual)

        self.calendar = calendar if calendar is not None else ql.NullCalendar()
        convention = ql.Unadjusted
        maturity_date_convention = convention
        rule = ql.DateGeneration.Backward
        coupons = ql.DoubleVector(1)
        coupons[0] = coupon / 100.0
        payment_convention = ql.ModifiedFollowing
        end_of_month = True

        schedule = ql.Schedule(self.issue_date, self.maturity_date,
                               tenor, self.calendar, convention, maturity_date_convention,
                               rule, end_of_month)
        self.accrualDayCounter = ql.ActualActual(ql.ActualActual.Bond, schedule)
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

    def _set_pricing_engine(self, bond_yield):
        term_structure = ql.YieldTermStructureHandle(
            ql.FlatForward(
                self.settlementDays,
                self.calendar,
                bond_yield,
                self.accrualDayCounter,
                self.compounding,
                ql.Annual))

        term_structure.enableExtrapolation()
        engine = ql.DiscountingBondEngine(term_structure)
        self.bond.setPricingEngine(engine)

    def clean_price(self, bond_yield, eval_date=None):
        if eval_date:
            self.set_evaluation_date(eval_date)
        self._set_pricing_engine(bond_yield)
        prc = self.bond.cleanPrice(bond_yield, ql.ActualActual(ql.ActualActual.Bond), ql.Compounded, ql.Annual)
        return prc

    def dirty_price(self, bond_yield, eval_date=None):
        if eval_date:
            self.set_evaluation_date(eval_date)
        self._set_pricing_engine(bond_yield)
        return self.bond.dirtyPrice()

    def accrued_amount(self, bond_yield):
        self._set_pricing_engine(bond_yield)
        return self.bond.accruedAmount()

    def duration(self, bond_yield, kind=ql.Duration.Simple):
        return ql.BondFunctions.duration(
            self.bond,
            bond_yield,
            self.accrualDayCounter,
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
            self.accrualDayCounter,
            self.compounding,
            self.frequency)

    def bps(self, bond_yield):
        return ql.BondFunctions.basisPointValue(
            self.bond,
            bond_yield,
            self.accrualDayCounter,
            self.compounding,
            self.frequency)

    def years_to_maturity(self, eval_date=None):
        eval_date = self._date_to_quantlib(dt.date.today()) if eval_date is None else self._date_to_quantlib(eval_date)
        maturity = self.maturity_date
        ytm = (maturity - eval_date) / 365.25
        return ytm

    def bond_yield(self, price, eval_date=None):
        if eval_date:
            self.set_evaluation_date(eval_date)
        yld = self.bond.bondYield(price, self.accrualDayCounter, self.compounding, ql.Annual)
        return yld

    def futures_contract_conversion_factor(self, bond_price, futures_price, futures_delivery_date, futures_coupon=0.06):
        """
        The Conversion Factor for a cash Treasury security is the price of that security that would makes
        its yield to the futures delivery date equal to its coupon rate.
        """
        cf = self.clean_price(bond_yield=futures_coupon, eval_date=futures_delivery_date) / 100
        adjusted_futures_price = futures_price * cf
        basis = bond_price - adjusted_futures_price
        return cf, basis


if __name__ == '__main__':

    # A día 2022-04-01
    date = ql.Date(1, 4, 2022)
    basket = [(0.90, ql.Date(1, 4, 2031), ql.Date(1, 10, 2020), 90.81),  # IT0005422891  ->  0.659620
              (0.60, ql.Date(1, 8, 2031), ql.Date(23, 2, 2021), 88.5),  # IT0005436693  ->  0.628867
              (0.95, ql.Date(1, 12, 2031), ql.Date(1, 6, 2021), 91.0),  # IT0005449969  ->  0.643888
              (0.95, ql.Date(1, 6, 2032), ql.Date(1, 11, 2021), 90.09)  # IT0005466013  ->  0.630012
              ]
    f_price = 137.54
    f_delivery = ql.Date(10,6,2022)

    securities = []
    min_basis = 100
    min_basis_index = -1
    for i, b in enumerate(basket):
        coupon, maturity, issue, price = b
        s = FixedRateBond(issue, maturity, coupon, settlement_days=0, calendar=ql.Germany())
        cf, basis = s.futures_contract_conversion_factor(bond_price=price, futures_price=f_price,
                                                         futures_delivery_date=f_delivery)
        if basis < min_basis:
            min_basis = basis
            min_basis_index = i
        securities.append((s, cf))

    ctd_info = basket[min_basis_index]
    ctd_bond, ctd_cf = securities[min_basis_index]
    ctd_price = ctd_info[3]

    print("%-30s = %lf" % ("Minimum Basis", min_basis))
    print("%-30s = %lf" % ("Conversion Factor", ctd_cf))
    print("%-30s = %lf" % ("Coupon", ctd_info[0]))
    print("%-30s = %s" % ("Maturity", ctd_info[1]))
    print("%-30s = %lf" % ("Price", ctd_info[3]))
