"""Contains models used to store clustering data.
"""

#   Stdlib
import os
import time
import csv
import datetime
import collections
import pdb
import logging
import traceback
import json
from operator import itemgetter
try:
    import cPickle as pickle
except ImportError:
    import pickle
from pprint import pformat

#   3rd party
from termcolor import colored
import pandas as pd
from pandas import DataFrame, Series
#       Numpy/scipy
import numpy as np
from scipy import linalg

#   Custom external
# from textclean.spacy_processor import SpacyProcessor
from textclean.v1.util import STOPWORDS
from textclean.v1 import (StandardizeText, NormalizeSpecialChars,
                          NormalizeWordLength, RemoveStopwords,
                          TokenizeToWords)
from textclean.v1.util import Pipeline

#   Custom current
from .util import mproperty
from .exceptions import NoDataError
from . import conf


LOGGER = logging.getLogger(__name__)


# Spacy processor
# SPACY_PROCESSOR = SpacyProcessor()




