
#   Stdlib
import os
import random
import logging
import pdb
from collections import Counter, deque, OrderedDict
from pprint import pformat



class LRUCache(object):
    def __init__(self, data=None, max_size=10000):
        if data:
            #   `data` is a list of (k, v) tuples.
            data_ = OrderedDict()
            for k, v in data:
                data_[k] = v
            data = data_
        else:
            data = OrderedDict()
        self.data = data
        self.freq = Counter()
        self.hits = 0.0
        self.misses = 0.0
        self.max_size = max_size or 10000
        self.size = 0

    def __getitem__(self, key):
        #   Pop and reinsert to refresh order.
        val = self.data.pop(key)
        self.data[key] = val
        self.freq[key] += 1
        #   Record a hit.
        self.hits += 1
        return val

    def __setitem__(self, key, val):
        if key in self.data:
            #   Pop and reinsert to refresh order.
            del self.data[key]
            self.size -= 1
        else:
            self.resize()
        self.data[key] = val
        self.size += 1
        self.freq[key] += 1

    def __delitem__(self, key):
        if key in self.data:
            del self.data[key]
            self.freq[key] -= 1
            if self.freq[key] < 1:
                del self.freq[key]
            self.size -= 1

    def __contains__(self, key):
        return key in self.data

    def get(self, key, dflt=None):
        if key in self.data:
            val = self.__getitem__(key)
        else:
            val = dflt
            #   Record a miss
            self.misses += 1
        return val

    def setdefault(self, key, val):
        if key in self.data:
            val = self.__getitem__(key)
        else:
            self.misses += 1
            self.__setitem__(key, val)
        return val

    def resize(self):
        while self.size > self.max_size:
            #   Pop the oldest item.
            k, v = self.data.popitem(last=False)
            ###########
            # print self.size, k, v
            # pdb.set_trace()
            ###########
            if self.freq[k] > 1:
                #   Possibly frequent. Reinsert to give item another chance.
                self.data[k] = v
                #   Reduce frequency.
                self.freq[k] = self.freq[k] / 10
            else:
                self.size -= 1
                del self.freq[k]

    def popitem(self):
        k, v = self.data.popitem(last=False)
        self.size -= 1
        del self.freq[k]
        return k, v

    def get_ratio(self):
        return self.hits / (self.hits + self.misses)
