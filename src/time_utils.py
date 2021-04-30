#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# Brief: a wrapper for the time counter.

import time


def timeit(f):

    def wrap(*args, **kargs):
        time1 = time.time()
        ret = f(*args, **kargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__,
                                                    (time2 - time1) * 1000.0))

        return ret

    return wrap
