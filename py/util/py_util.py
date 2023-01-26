import datetime


def timed_print(s, *args, **kwargs):
    t = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    print(f'{t} {s}', *args, **kwargs)
