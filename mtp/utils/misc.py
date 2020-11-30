import sys
from functools import wraps
from time import perf_counter

from utils.logging import get_logger

logger = get_logger(__name__)


def timer(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        resp = f(*args, **kwargs)
        end = perf_counter()
        logger.info('{}, {}'.format(f.__name__, end - start))
        return resp
    return wrapper


_intervals = (
    ('weeks', 604800, int),  # 60 * 60 * 24 * 7
    ('days', 86400, int),    # 60 * 60 * 24
    ('hours', 3600, int),    # 60 * 60
    ('minutes', 60, int),
    ('seconds', 1.0, float),
    )


def display_time(seconds):
    result = []

    for name, count, dtype in _intervals:
        if dtype == float:
            value = seconds / count
        else:
            value = seconds // count
        if value:
            seconds -= value * count
            result.append('{:02d}'.format(int(round(value))))
    return ':'.join(result)


def pool_with_progress(pool, func_process, item_list):
    t1 = perf_counter()
    vl = []
    for i, v in enumerate(pool.imap_unordered(func_process, item_list), 1):
        vl.append(v)
        if (i + 1) % 10 == 0:
            t2 = perf_counter()
            elapsed_time = t2 - t1
            progress = (i + 1) / len(item_list)
            total_estimated_time = 1 / progress * elapsed_time
            estimated_finished_time = total_estimated_time - elapsed_time
            result = ['elapsed time: {}'.format(display_time(elapsed_time))] + \
                     ['rest: {}'.format(display_time(estimated_finished_time))] + \
                     ['done {:6.4f}%'.format((i + 1) / len(item_list) * 100)]
            sys.stderr.write('\r{}'.format(', '.join(result)))
            sys.stderr.flush()
    return vl
