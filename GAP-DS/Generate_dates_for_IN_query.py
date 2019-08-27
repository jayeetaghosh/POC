from datetime import datetime, timedelta

from_date = datetime(2017, 9, 1)
to_date = datetime(2017, 12, 31)

def get_dates(from_date, to_date):
    """
    Generates dates in string format (eg: 05-SEP-17) for all dates in the closed 
    interval [from_date, to_date]
    """
    dt = from_date
    while dt <= to_date:
        yield dt.strftime('%d-%b-%y').upper()
        dt += timedelta(days=1)

dates = get_dates(from_date, to_date)
s = '(' + ','.join("'{}'".format(x) for x in dates) + ')'
print(s)
