import numpy as np
from scipy.optimize import minimize
import datetime as dt
from MachineTrading.Utils.date_utils import date_to_quantlib


def nelson_siegel_rate(term, tau, b0, b1, b2):
    x = term / tau
    r = b0 + (b1 + b2) * (1 - np.exp(-x)) / x - b2 * np.exp(-x)
    return r


def nelson_siegel_discount_factor(term, tau, b0, b1, b2):
    df = np.exp(-nelson_siegel_rate(term, tau, b0, b1, b2) * term)
    return df


def nelson_siegel_bond_price(b, tau, b0, b1, b2, eval_date):
    if isinstance(eval_date, dt.date):
        eval_date = date_to_quantlib(eval_date)

    cashflows = b.bond.bond.cashflows()
    cf = [(x.date(), x.amount() / 100, b.bond.pricingDayCounter.yearFraction(eval_date, x.date()))
          for x in cashflows]
    cf = [x for x in cf if x[2] > 0]

    coupons = np.array([x[1] for x in cf])
    terms = np.array([x[2] for x in cf])

    cpns = 100 * (coupons[:-1] * nelson_siegel_discount_factor(terms[:-1], tau, b0, b1, b2))
    face = 100 * (nelson_siegel_discount_factor(terms[-1], tau, b0, b1, b2))
    price = np.sum(cpns) + face
    return price


def nelson_siegel_price_error(b, tau, b0, b1, b2, eval_date):
    ns_price = nelson_siegel_bond_price(b, tau, b0, b1, b2, eval_date)
    price_error = (ns_price - b.price) ** 2
    return price_error


def nelson_siegel_curve_fit(bonds, eval_date, disp=False):
    def fitting_func(args):
        tau, b0, b1, b2 = args
        total_error = 0
        for isin, b in bonds.items():
            price_error = nelson_siegel_price_error(b, tau, b0, b1, b2, eval_date)
            total_error += price_error
        return np.sqrt(total_error)

    ns_minimization_result = minimize(fitting_func,
                                      x0=np.array([1, 0, 0, 0]),
                                      bounds=[(0, None), (0, 5), (-5, 5), (-5, 5)],
                                      tol=1e-8,
                                      options={'disp': disp}
                                      )
    return ns_minimization_result
