

#   Stdlib
import time
import datetime
import copy
import os
import pdb  #   Debugger
import string
import random
import operator
import csv
import logging
from collections import defaultdict, deque
from pprint import pformat
try:
    import regex as re
except ImportError:
    import re

#   3rd party
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from tables import IsDescription, StringCol, open_file
from termcolor import colored, cprint
from pymongo import MongoClient
from dateutil.parser import parse as date_parse

#   Custom
from cluster_webapp import conf
from cluster_webapp.lib.clustering import clusterers

#   Custom external
from textclean.v1 import clean as text_clean

#: Logger name
LOGGER = logging.getLogger("base.models.base")


def maybe_date_parse(x):
    if isinstance(x, basestring): x = date_parse(x)
    return x
vec_maybe_date_parse = np.vectorize(maybe_date_parse)

def maybe_join_seq(x):
    if hasattr(x, '__iter__'): x = u'|'.join(x)
    return x
vec_maybe_join_seq = np.vectorize(maybe_join_seq)

def maybe_join_seq_comma(x):
    if hasattr(x, '__iter__'): x = u', '.join(x)
    return x
vec_maybe_join_seq_comma = np.vectorize(maybe_join_seq_comma)

def maybe_encode_utf8(x):
    if isinstance(x, unicode): x = x.encode('utf-8')
    return x
vec_maybe_encode_utf8 = np.vectorize(maybe_encode_utf8)


class ClusterDataFrame(object):
    """
    Wraps a DataFrame containing cluster data.

    """
    #: Base directory of raw data.
    raw_base_path = conf.RAW_DATA_PATH
    if not os.path.exists(raw_base_path): os.makedirs(raw_base_path)
    #: Base directory of normalized data.
    clean_base_path = conf.CLEAN_DATA_PATH
    if not os.path.exists(clean_base_path): os.makedirs(clean_base_path)
    #: Base directory of output data.
    final_base_path = conf.FINAL_DATA_PATH
    if not os.path.exists(final_base_path): os.makedirs(final_base_path)
    #: Base directory of metadata.
    meta_base_path = conf.META_DATA_PATH
    if not os.path.exists(meta_base_path): os.makedirs(meta_base_path)
    #: Default columns to include in data after calling :py:meth:`.load`
    dflt_columns = [
        '_id',
        'cluster_id',
        'clean_content',
        'clean_tokens',
    ]
    text_cols = keep_cols = drop_cols = info_cols = []
    info_fields = []
    #: Name of field containing datetime information in the source data.
    fld_date = 'date_create'

    def __init__(self, fname, refresh=False, columns=None, db_name=None, **kwargs):
        """
        Initialize DataFrame containing cluster data.

        :param str fname: Name of file with data
        :param bool refresh: Whether to overwrite existing data (Currently always True)
        :param list columns: Column names (defaults to :py:attr:`.dflt_columns`)
        :param str db_name: Database name
        :param dict kwargs: Misc kwargs

        :return: None

        """
        if columns is None: columns = None  #   self.dflt_columns
        self.columns = columns
        self.fname = fname
        self.refresh = True #refresh
        self.loaded = False
        self.df = None
        self.db_name = db_name

    @property
    def clean_path(self):
        """
        Location of normalized data. Combination of :py:attr:`.clean_base_path` and :py:attr:`.fname`

        """
        try:
            self.__clean_path
        except AttributeError:
            self.__clean_path = os.path.join(self.clean_base_path, self.fname)
        return self.__clean_path

    @property
    def raw_path(self):
        """
        Location of raw data. Combination of :py:attr:`.raw_base_path` and :py:attr:`.fname`

        """
        try:
            self.__raw_path
        except AttributeError:
            self.__raw_path = os.path.join(self.raw_base_path, self.fname)
        return self.__raw_path

    @property
    def final_path(self):
        """Location of user-facing output. Combination of :py:attr:`.final_base_path` and :py:attr:`.fname`

        """
        try:
            self.__final_path
        except AttributeError:
            self.__final_path = os.path.join(self.final_base_path, self.fname)
        return self.__final_path

    @property
    def meta_path(self):
        """Metadata location. Combination of :py:attr:`.meta_base_path` and :py:attr:`.fname`

        """
        try:
            self.__meta_path
        except AttributeError:
            self.__meta_path = os.path.join(self.meta_base_path, self.fname)
        return self.__meta_path

    def load(self, to_csv=True, limit=None):
        """
        Load a csv file and convert to a dataframe.

        (1) If :py:attr`.refresh` is True or the normalized data cannot be found, load the raw data and normalize it. Otherwise, load the existing normalized data.
        (2) Save the normalized data to csv if 'to_csv' is :py:obj:`True`.
        (3) Restrict the dataframe to columns specified in :py:attr`.columns`.
        (4) Restrict the dataframe to the number of rows specified in 'limit'.
        (5) Set :py:attr:`df` to the resulting dataframe.

        :param bool to_csv: Whether to save the dataframe to a csv file (defaults to True).
        :param int limit: Number of lines to read (defaults to no limit).

        :return: Dataframe loaded from csv file.
        :rtype: pd.DataFrame

        """
        clean_path = self.clean_path
        raw_path = self.raw_path
        if (not self.refresh) and os.path.exists(clean_path):
            df = pd.read_csv(clean_path)
        else:
            df = pd.read_csv(raw_path)
            df = self.normalize(df)
            #   Save to csv
            if to_csv:
                self.to_csv(df, index=False)
        if self.columns is not None:
            #   Keep only relevant columns.
            columns = [ c for c in self.columns if c in df ]
            df = df[columns]
        if limit:
            df = df.iloc[:limit]
        self.df = df
        self.loaded = True
        return df

    def create_clean_content(self, df):
        """
        Clean dataframe content by calling :py:func:`text_clean.standardize_text` on each row.

        :param pd.DataFrame df: Dataframe to clean.

        :return: Cleaned Dataframe
        :rtype: pd.DataFrame

        """
        return df.content.map(text_clean.standardize_text)

    def normalize_clean_content(self, df):
        """
        Further normalize cleaned dataframe content.

        :param pd.DataFrame df: Dataframe to normalize.

        :return: Normalized Dataframe
        :rtype: pd.DataFrame

        """
        if 'clean_content' not in df:
            df['clean_content'] = self.create_clean_content(df)
            #   Remove the content column.
            df.drop('content', axis=1)
        #   Ensure na items are text.
        df.clean_content = df.clean_content.fillna('')
        #   Truncate all whitespace.
        df.clean_content = df.clean_content.str.replace(r'\s+', r' ')
        #   Encode data to utf-8
        df.clean_content = df.clean_content.map(maybe_encode_utf8)
        return df

    def create_clean_tokens(self, df):
        """
        Create tokens from cleaned content.

        :param pd.DataFrame df: Dataframe to tokenize.

        :return: Tokenized content
        :rtype: pd.Series

        """
        clean_tokens = df.clean_content.map(
            text_clean.simplify_text).map(
                operator.itemgetter(1)
        )
        #   Remove numbers from the tokens
        clean_tokens = clean_tokens.map(self.clean_tokens)
        return clean_tokens

    def normalize_clean_tokens(self, df):
        """
        Further normalize tokenized content.

        :param pd.DataFrame df: Dataframe to normalize.

        :return: Dataframe with tokenized content
        :rtype: pd.Dataframe

        """
        if 'clean_tokens' not in df:
            df['clean_tokens'] = self.create_clean_tokens(df)
        #   Remove null values from the clean_tokens.
        df = df.dropna(subset=['clean_tokens'])
        #   Cast each list of clean tokens to a string, if necessary
        df.clean_tokens = df.clean_tokens.map(maybe_join_seq)
        #   Encode data to utf-8
        df.clean_tokens = df.clean_tokens.fillna('')
        df.clean_tokens = df.clean_tokens.map(maybe_encode_utf8)
        return df

    def normalize_labels(self, df):
        """
        Join labels with commas.
        """
        if 'label' in df:
            df.label = df.label.map(maybe_join_seq_comma)
        return df

    def normalize_text(self, df):
        """
        Normalize dataframe text by calling :py:meth:`normalize_clean_content`, :py:meth:`normalize_clean_tokens` and :py:meth:`normalize_labels` on the dataframe, and finally encoding all text columns to utf-8.

        """
        df = self.normalize_clean_content(df)
        df = self.normalize_clean_tokens(df)
        df = self.normalize_labels(df)
        for col in self.text_cols:
            if col in df:
                df[col] = df[col].fillna('').map(maybe_encode_utf8)
        return df

    def normalize_ids(self, df):
        """
        Normalize ids by creating the '_id' field if not present and dropping duplicates.

        """
        if '_id' not in df: df['_id'] = xrange(len(df))
        #   Remove duplicates
        df = df.drop_duplicates(cols=['_id'])
        return df

    def normalize_dates(self, df):
        """
        Normalize dates by parsing the dates, renaming the date field to 'datetime', and stringifying the entries of the 'time', 'date' and 'datetime' columns.

        """
        fld_date = self.fld_date
        try:
            if fld_date in df:
                df[fld_date] = df[fld_date].map(maybe_date_parse)
                df['datetime'] = df[fld_date]
                df = df.drop([fld_date], axis=1)
                df['time'] = df['datetime'].map(
                    lambda dt: dt.strftime('%H:%M:%S')
                )
                df['date'] = df['datetime'].map(
                    lambda dt: dt.strftime('%m/%d/%Y')
                )
                df['datetime'] = df['datetime'].map(
                    lambda dt: dt.isoformat()
                )
        except Exception as e:
            LOGGER.error(e)
            LOGGER.error(df)
            # pdb.set_trace()
        return df

    def normalize(self, df, to_csv=False):
        """
        Normalize *ids*, *dates* and *text* of a dataframe.

        Required fields in input dataframe:

            * ``content``

        """
        if not (df.empty or getattr(df, 'is_normalized', False)):
            df = df.copy()
            #   Format the dates
            df = self.normalize_ids(df)
            df = self.normalize_dates(df)
            df = self.normalize_text(df)
            if self.db_name: df.db_name = self.db_name
            df.is_normalized = True
        return df

    num_regex = re.compile('\d+')
    @classmethod
    def clean_tokens(cls, tokens):
        """
        Remove tokens containing numbers.

        """
        num_regex = cls.num_regex
        tokens = [ w for w in tokens if not num_regex.match(w) ]
        return tokens

    def to_h5(self):
        """
        Save :py:attr:`.df` to pytables.

        (1) Normalizes :py:attr:`.df` with :py:meth:`.normalize`
        (2) Creates a new :py:class:`ClusterTable` instance and calls :py:meth:`ClusterTable.create_groups`

        :return: The list of children for the table.
        :rtype: list

        """
        df = self.df
        df = self.normalize(df)
        #   Put the data into the format expected by the h5 storage routines
        df = df.rename(columns={'_id': 'id'})
        name = (
            os.path.splitext(self.fname)[0] +
            datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        )
        if not 'cluster_id' in df: df['cluster_id'] = name
        h5fname = name + '.h5'
        table = ClusterTable(
            df,
            name=name,
            parent_path='/',
            h5fname=h5fname,
        )
        children = table.create_groups()
        children = [
            {
                'sample'        : sample,
                'group_path'    : path,
                'table_name'    : name,
                'h5fname'       : h5fname,
                'db_name'       : self.db_name,
                'size'          : size,
                'tags'          : tags,
                'info_feats'    : info_feats,
                'id'            : ClusterTable.create_id(h5fname, path, name),
            }
                for sample, path, name, size, tags, info_feats in children
        ]
        return children

    def to_csv(self, df, path=None, **kwargs_csv):
        """
        Saves a dataframe to csv.

        """
        if path is None: path = self.clean_path
        #   Output df index by default.
        kwargs_csv.setdefault('index', True)
        #   ------------------------------------
        LOGGER.info('Saving data in clustering format to {0}.'.format(
            colored(path, 'red')
        ))
        #   ------------------------------------
        df.to_csv(path, **kwargs_csv)

    @staticmethod
    def sample(df, n=50):
        """
        Samples a dataframe

        :param pd.DataFrame df: Dataframe to sample
        :param int n: Number of items (defaults to 50)

        """
        size = len(df)
        if n < size:
            idxs = random.sample(xrange(size), n)
            samples= df.iloc[idxs]
        else:
            samples = df
        return samples

    @staticmethod
    def ravel_labels(df):
        """
        (Un)ravels a dataframe: Creates a new row for each token in the column 'labels' for each row.

        :param pd.DataFrame df: Dataframe to unravel.

        :return: The unraveled dataframe
        :rtype: pd.DataFrame

        """
        if any(df.label.map(lambda x: isinstance(x, basestring))):
            df.label = df.label.map(lambda x: x.split(','))
        rows = []
        for k, row in df.iterrows():
            d = row.to_dict()
            labels = d['label']
            #   Normalize and de-dup the labels.
            labels = [t.lower().strip() for t in labels]
            labels = set(labels)
            #   Add a new row for each label.
            for label in labels:
                new_d = dict(d)
                new_d['label'] = label
                rows.append(new_d)
        new_df = DataFrame(rows)
        return new_df

    @staticmethod
    def ravel_label_row(d):
        """
        Creates a new row for each label in the data.

        :param dict d: Dictionary representing a DataFrame row.

        :return: Generator of new entries
        :rtype: Generator

        """
        labels = d.pop('label', []) or d.pop('labels', [])
        #   Normalize and de-dup the labels.
        labels = [t.lower().strip() for t in labels]
        labels = set(labels)
        #   Add a new row for each label.
        for label in labels:
            new_d = dict(d)
            new_d['label'] = label
            yield new_d

