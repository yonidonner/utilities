import functools

import numpy

def make_cached_queue_reader(q, size, rnd=None):
    if rnd is None:
        rnd = numpy.random.RandomState()
    def queue_get():
        item = q.get()
        q.task_done()
        return item
    cache = []
    counts = numpy.zeros([size], dtype=numpy.int32)
    choose = lambda x: x[rnd.randint(0, len(x))]
    choose_min = choose((counts == counts.min()).nonzero()[0])
    choose_max = choose((counts == counts.max()).nonzero()[0])
    def get():
        if len(cache) < size:
            ret_item = queue_get()
            counts[len(cache)] = 1
            cache.append(ret_item)
            return ret_item
        if q.empty():
            ind = choose_min()
            counts[ind] += 1
            return cache[ind]
        else:
            ind = choose_max()
            ret_item = queue_get()
            counts[ind] = 1
            cache[ind] = ret_item
            return ret_item
    return get
