import datetime as dt
import pandas as pd
from Utils import io_utils
from Instruments.fixed_rate import FixedRateBond


def build_bond(isin):
    bond_info = io_utils.read_json(f"Data/BondInfo/{isin}.json")
    yield_data = io_utils.investing_yield_data(f"Data/Yields/{isin}.csv")
    maturity = dt.datetime.strptime(bond_info["MaturityDate"], '%Y-%m-%d')
    issue = dt.datetime.strptime(bond_info["IssueDate"], '%Y-%m-%d')
    coupon = bond_info["CouponInterest"]
    bond = FixedRateBond(issue_date=issue, maturity_date=maturity, coupon=coupon, settlement_days=2)
    bond.yield_data = yield_data
    return bond


if __name__ == '__main__':
    test = build_bond("IT0005466013")
