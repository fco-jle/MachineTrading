import QuantLib as ql
import prettytable as pt
import datetime as dt


class FixedRateBond:
    def __init__(self, issue_date, maturity_date, coupon, settlement_days=1):
        self.settlementDays = settlement_days
        face_amount = 100.0
        self.issue_date = self._date_to_quantlib(issue_date)
        self.maturity_date = self._date_to_quantlib(maturity_date)
        self.frequency = ql.Semiannual
        self.compounding = ql.Compounded
        tenor = ql.Period(ql.Semiannual)
        self.calendar = ql.Italy()
        convention = ql.Unadjusted
        maturity_date_convention = convention
        rule = ql.DateGeneration.Backward

        end_of_month = False
        schedule = ql.Schedule(
            self.issue_date,
            self.maturity_date,
            tenor,
            self.calendar,
            convention,
            maturity_date_convention,
            rule,
            end_of_month)

        coupons = ql.DoubleVector(1)
        coupons[0] = coupon / 100.0
        self.accrualDayCounter = ql.ActualActual(ql.ActualActual.Bond, schedule)
        payment_convention = ql.Unadjusted

        self.bond = ql.FixedRateBond(
            self.settlementDays,
            face_amount,
            schedule,
            coupons,
            self.accrualDayCounter,
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
                self.frequency))
        term_structure.enableExtrapolation()
        engine = ql.DiscountingBondEngine(term_structure)
        self.bond.setPricingEngine(engine)

    def clean_price(self, bond_yield):
        self._set_pricing_engine(bond_yield)
        return self.bond.cleanPrice()

    def dirty_price(self, bond_yield):
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


if __name__ == '__main__':
    example = FixedRateBond(
        issue_date=ql.Date(1, 3, 2015),
        maturity_date=ql.Date(1, 3, 2032),
        coupon=1.650,
        settlement_days=1)

    ex_yield = 0.0179

    example.cash_flows()
    c_price = example.clean_price(ex_yield)
    d_price = example.dirty_price(ex_yield)
    acc_amount = example.accrued_amount(ex_yield)

    dur_simp = example.duration_simple(ex_yield)
    dur_mod = example.duration_modified(ex_yield)
    dur_mac = example.duration_macaulay(ex_yield)

    tabDuration = pt.PrettyTable(['Item', 'Value'])
    tabDuration.add_row(['Clean Price', c_price])
    tabDuration.add_row(['Dirty Price', d_price])
    tabDuration.add_row(['Accrued Amount', acc_amount])
    tabDuration.add_row(['Duration Simple', dur_simp])
    tabDuration.add_row(['Duration Modified', dur_mod])
    tabDuration.add_row(['Duration Macaulay', dur_mac])
