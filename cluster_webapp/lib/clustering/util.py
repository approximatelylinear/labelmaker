
"""
Internal utilities for the clustering package
"""

#   Stdlib
import csv
import itertools
import os
import logging
import json
from functools import wraps

LOGGER = logging.getLogger(__name__)

#:  The directory containing this script
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_DATA = os.path.join(THIS_DIR, 'data')


def mproperty(fn):
    """
    Memoize a class property definition

    class Foo(object):
        @mproperty
        def bar(self):
            return 2 * 2 * 2
    """
    attribute = "_{}".format(fn.__name__)

    @property
    @wraps(fn)
    def _property(self):
        try:
            val = getattr(self, attribute)
        except AttributeError:
            val = fn(self)
            setattr(self, attribute, val)
        return val

    return _property


def csv_2_json(fname, length=None, tweet_type=None):
    """Transforms a CSV file into JSON format.
    """
    if not fname.startswith(PATH_DATA): fname = os.path.join(PATH_DATA, fname)
    with open(fname, mode='rbU') as f:
        reader = csv.DictReader(f)
        rows = (r for r in reader)
        if length:
            rows = itertools.islice(rows, 0, length)
        if tweet_type:
            rows = (r for r in rows if r['type'] == tweet_type)
        rows = list(rows)
    json_fname = os.path.splitext(fname)[0] + '.json'
    ################
    msg = 'Serializing results to JSON and writing to {}...'.format(
        json_fname),
    LOGGER.info(msg)
    ################
    try:
        with open(json_fname, mode='wb') as f:
            json.dump(rows, f, indent=' ' * 4)
        msg = 'Finished!'
        LOGGER.info(msg)
    except Exception as exc:
        LOGGER.exception('Failed: {}'.format(exc))
