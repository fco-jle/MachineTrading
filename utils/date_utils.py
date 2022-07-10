import QuantLib as ql
import datetime as dt


def date_to_quantlib(date):
    if isinstance(date, dt.date):
        return ql.Date(date.day, date.month, date.year)
    elif isinstance(date, ql.Date):
        return date
    else:
        raise ValueError

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + dt.timedelta(n)