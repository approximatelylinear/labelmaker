

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
from cluster_webapp.lib.clustering import clusterers
from cluster_webapp.lib.text_processing import clean as text_clean

#: The directory containing this script
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

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
    raw_base_path = os.path.join(THIS_DIR, 'data', 'raw')
    if not os.path.exists(raw_base_path): os.makedirs(raw_base_path)
    #: Base directory of normalized data.
    clean_base_path = os.path.join(THIS_DIR, 'data', 'clean')
    if not os.path.exists(clean_base_path): os.makedirs(clean_base_path)
    #: Base directory of output data.
    final_base_path = os.path.join(THIS_DIR, 'data', 'final')
    if not os.path.exists(final_base_path): os.makedirs(final_base_path)
    #: Base directory of metadata.
    meta_base_path = os.path.join(THIS_DIR, 'data', 'meta')
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
            ######
            # pdb.set_trace()
            ######
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

    def get_is_labeled_filter(self, is_labeled, query=None):
        """
        Filter items to those that have or haven't been labeled.

        """
        if query is None: query = {}
        if is_labeled is True:
            query['label_count'] = {'$gt': 0}
            # query['labels'] = {'$not': {'$size': 0}}
        elif is_labeled is False:
            query['label_count'] = 0
            # query['$or'] = [
            #     { 'labels'  : {'$exists': False } },
            #     { 'labels'  : { '$size' : 0 } },
            #     { 'labels'  : '__none__' }
            # ]
        return query

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


class Cluster(object):
    base_path = os.path.join(THIS_DIR, 'data', 'clusters')
    if not os.path.exists(base_path): os.makedirs(base_path)
    cluster_doc_dir = clusterers.CLUSTER_DOC_DIR
    cluster_columns = [
        'clean_content',
        'clean_tokens',
        'cluster_id',
        '_id'
    ]

    def __init__(
            self,
            table_name,
            group_path,
            num_clusters=2,
            refresh=False,
            cluster_method='sign',
            h5fname="clusters1.h5",
            db_name=None,
        ):
        self.table_name = table_name
        self.group_path = group_path
        self.h5fname = h5fname
        self.refresh = True #refresh
        self.num_clusters = num_clusters
        self.db_name = db_name
        self.clusterer = self.create_clusterer(cluster_method)
        self.df = None
        self.info_features = None
        self.now = datetime.datetime.now().strftime('%s')[:-4]

    def create_clusterer(self, name):
        if name == 'kmeans':
            clusterer = clusterers.KMeansClusterer(
                self.table_name, num_clusters=self.num_clusters,
                save=False, refresh=self.refresh
            )
        else:
            clusterer = clusterers.SignClusterer(
                self.table_name, num_clusters=self.num_clusters,
                save=False, refresh=self.refresh
            )
        return clusterer

    def load(self):
        if self.df is None:
            #   Load the cluster data
            os_path = os.path.join(self.base_path, self.h5fname)
            with open_file(os_path, mode="r+") as h5file:
                #######
                # pdb.set_trace()
                #######
                cluster_path = '/'.join([self.group_path, self.table_name])
                cluster_table = h5file.get_node(cluster_path)
                df = ClusterTable.table_to_df(cluster_table)
                self.df = df
        ########
        # LOGGER.debug(self.df)
        # pdb.set_trace()
        ########
        return self.df

    def cluster(self, df=None, to_csv=True):
        func_name = '.'.join([__name__, self.__class__.__name__, 'cluster'])
        if df is None: df = self.load()
        #   Put the data into the format expected by the clustering routine
        #       Preserve the original format.
        orig_df = df
        df = df.rename(columns={'id': '_id'})
        cluster_columns = self.cluster_columns
        cluster_columns = [ c for c in cluster_columns if c in df ]
        df = df[cluster_columns]
        #   Save the content to the clustering directory
        # fname = self.table_name
        # if not fname.endswith('.pkl'): fname += '.pkl'
        # cluster_path = os.path.join(self.cluster_doc_dir, fname)
        # #   Save data to the location expected by the clustering routine.
        # df.to_pickle(cluster_path)
        ########
        # pdb.set_trace()
        ########
        clusterer = self.clusterer
        if len(df) > 2:
            #   --------------------------------------------
            LOGGER.info('Creating new clusters for {0}...'.format(colored(self.table_name, 'yellow')))
            #   --------------------------------------------
            try:
                #   If refresh, remove all saved data.
                clusterer.create_models(df)
                clusters = clusterer.doc_cluster(df, base_dir=self.base_path)
                info_feats = self.get_info_features()
                ##########
                # LOGGER.debug(clusters)
                # LOGGER.debug(info_feats)
                # pdb.set_trace()
                ##########
                #   Remove saved data.
                self.clusterer.delete_dependencies()

            except (ValueError, Exception) as e:
                #   ---------------------------------------------
                LOGGER.error('{n}: Error clustering:\n\t{e}'.format(
                    n=colored(func_name, 'yellow'), e=colored(e, 'red')
                ))
                clusters = df
                df['docnum'] = None
                df['cluster_id'] = 0
                #   ---------------------------------------------
        else:
            clusters = df
            df['docnum'] = None
            df['cluster_id'] = 0
        ########
        # LOGGER.debug(clusters)
        # pdb.set_trace()
        ########
        clusters = self.normalize(clusters)
        #   Add original columns back.
        for col in (set(orig_df) - set(clusters)):
            clusters[col] = orig_df[col]
        ########
        # LOGGER.debug(clusters)
        # pdb.set_trace()
        ########
        if to_csv: self.to_csv(clusters)
        return clusters

    def normalize(self, clusters):
        clusters['parent_id'] = self.table_name
        if any(clusters.clean_tokens.map(lambda x: isinstance(x, list))):
            clusters.clean_tokens = clusters.clean_tokens.map(lambda x: '|'.join(x))
        drop_cols = ['docnum']
        clusters = clusters.drop(drop_cols, axis=1)
        clusters = clusters.rename(columns={'_id': 'id'})
        clusters = self.set_cluster_ids(clusters)
        return clusters

    def set_cluster_ids(self, clusters):
        #   Add the 5 most sig digits of the current time in seconds to
        #   the cluster index in order to make cluster identifiers
        #   unique across runs of this routine.
        clusters.cluster_id = clusters.cluster_id.map(
            lambda x: '_'.join(['c' + str(int(x)), self.now])
        )
        return clusters

    def to_h5(self, clusters):
        table = ClusterTable(
            clusters,
            name=self.table_name,
            parent_path=self.group_path,
            h5fname=self.h5fname,
            info_feats=self.info_features,
        )
        children = table.create_groups()
        children = [
            {
                'sample'        : sample,
                'group_path'    : path,
                'table_name'    : name,
                'db_name'       : self.db_name,
                'h5fname'       : self.h5fname,
                'size'          : size,
                'tags'          : tags,
                'info_feats'    : info_feats,
                'id'            : '-'.join(
                    [
                        self.h5fname.replace('/', '-'),
                        path.replace('/', '-'),
                        name.replace('/', '-')
                    ]
                )
            }
                for sample, path, name, size, tags, info_feats in children
        ]
        return children

    def to_csv(self, clusters):
        #   Save the clusters with the query
        target_path = os.path.join(self.base_path, 'cluster_dfs')
        if not os.path.exists(target_path): os.makedirs(target_path)
        fname = '{0}.csv'.format(self.table_name)
        path = os.path.join(target_path, fname)
        #############
        # pdb.set_trace()
        #############
        try:
            clusters.to_csv(path, index=False)
        except Exception as e:
            print e
            #   Try to modify the encoding
            for col in clusters:
                pass
                # pdb.set_trace()

    def get_info_features(self, to_csv=False):
        func_name = '.'.join([__name__, self.__class__.__name__, 'get_info_features'])
        try:
            clusterer = self.clusterer
            #   Get features characterizing each cluster.
            info_features = clusterer.get_info_features()
            #######
            # pdb.set_trace()
            #######
            info_features = self.set_cluster_ids(info_features)
            info_features = info_features.rename(
                columns={'counts': 'magnitude', 'ids': 'words'}
            )
            # lvl_idxs = list(set(all_counts.index.get_level_values(0)))
            if to_csv:
                #   Save informative features.
                info_feats_path = os.path.join(base_path, 'info_feats')
                if not os.path.exists(info_feats_path): os.makedirs(info_feats_path)
                #   TODO:   Fix the use of `data_id`.
                info_feats_fname = '{0}_characteristic_words.csv'.format(data_id)
                info_feats_path = os.path.join(info_feats_path, fname)
                info_features.to_csv(info_feats_path, index=False)

            def flatten_info_feats(df):
                df = df.sort_index(by='magnitude', ascending=False)
                #   Return at most 10 features.
                df = df.iloc[:min(10, len(df))]
                data = [ r.to_dict() for k, r in df.iterrows() ]
                return data

            info_features = info_features.groupby('cluster_id')
            info_features = info_features.apply(flatten_info_feats)
            ##########
            # LOGGER.debug(info_features)
            # pdb.set_trace()
            ##########
        except Exception as e:
            #######################################################
            LOGGER.error('{n}: Error getting informative features:\n\t{e}'.format(
                n=colored(func_name, 'yellow'), e=colored(e, 'red')
            ))
            # pdb.set_trace()
            #######################################################
        self.info_features = info_features
        return info_features


class ClusterSchemaFactory(object):
    def __init__(self, df):
        self.df = df

    @property
    def id_size(self):
        return max(self.df.id.map(len))

    @property
    def cluster_id_size(self):
        return max(self.df.cluster_id.map(len))

    @property
    def clean_content_size(self):
        #   TBD: Change this so we aren't modifying data in a getter.
        self.df.clean_content = self.df.clean_content.fillna('')
        return max(self.df.clean_content.map(len))

    @property
    def clean_tokens_size(self):
        return max(self.df.clean_tokens.map(len))

    @property
    def misc_size(self):
        #   Serialize unsupported columns into the 'other' entry
        if 'misc' in self.df:
            misc_col = self.df.misc
        else:
            misc_col = None
        return max(misc_col.map(len)) if misc_col is not None else 1

    def __call__(self):
        class Cluster(IsDescription):
            id              = StringCol(self.id_size)
            clean_content   = StringCol(self.clean_content_size)
            clean_tokens    = StringCol(self.clean_tokens_size)
            cluster_id      = StringCol(self.cluster_id_size)
            misc            = StringCol(self.misc_size)
        return Cluster


class ClusterTable(object):
    """
        STRUCTURE
        =========
                 ROOT
                  |
                GROUP0
                  |
                GROUP1                      #   Multiple groups possible at this level.
              /       \
        table           GROUP2
                      /        \
                GROUP3  ... ... GROUP4      #   Multiple groups possible at this level.
                /   \           /   \
            table   GROUP5  table   GROUP6
                       |              |
                     table           table  #   Last level; no groups.
    """
    base_path = os.path.join(THIS_DIR, 'data', 'clusters')
    if not os.path.exists(base_path): os.makedirs(base_path)
    def __init__(
            self,
            df=None,
            name=None,
            parent_path='/',
            h5fname="clusters1.h5",
            h5fmode=None,
            info_feats=None
        ):
        if h5fmode is None: h5fmode = 'w' if parent_path == '/' else 'a'
        if df is not None:
            df.id = df.id.map(str)
            df.index = df.id
            self.df = df
        if df is not None and 'cluster_id' in df:
            self.cluster_ids = df.cluster_id.unique()
        else:
            self.cluster_ids = []
        #   Serialize unsupported columns into the 'misc' entry
        misc_col = self.serialize_misc_cols(df)
        df['misc'] = misc_col
        #   Default to the data id.
        if name is None:
            if df is not None and 'parent_id' in df:
                name = df.parent_id[0]
            else:
                name = h5fname[:-3]
        self.name = name.replace('-', '_')
        self.parent_path = parent_path
        self.h5fname = h5fname
        self.h5fmode = h5fmode
        self.info_feats=info_feats
        schema_factory = ClusterSchemaFactory(df)
        self.schema = schema_factory()

    @staticmethod
    def fill_series(series):
        if (series.dtype == 'float') or (series.dtype == 'int'):
            if np.all(np.isnan(series)):
                series = series.fillna('')
            else:
                series = series.fillna(0)
        else:
            series = series.fillna('')
        return series

    @staticmethod
    def fill_series_with_list(series):
        #   ==================================
        def has_iter(x):
            return hasattr(x, '__iter__')
        #   ==================================
        vec_has_iter = np.vectorize(has_iter)
        res = zip(vec_has_iter(series), series, [[]] * len(series))
        s = [ xv if c else yv for (c, xv, yv) in res ]
        series = Series(s, series.index)
        return series

    @staticmethod
    def serialize_misc_cols(df):
        """
        Serialize unsupported columns into the 'misc' entry
        """
        import json
        supported_cols = set([
            'clean_content',
            'clean_tokens',
            'id',
            'cluster_id',
            'parent_id',
            'labels'
        ])
        unsupported_cols = set(df.columns) - supported_cols
        other_col = None
        #   ==================================
        def get_vals(row):
            for col in unsupported_cols:
                v = row[col]
                # t = type(v)
                if isinstance(v, datetime.datetime):
                    v = v.isoformat()
                yield col, v

        def create_col_other():
            other = dict(
                (idx, json.dumps(dict(get_vals(row))))
                    for idx, row in df.iterrows()
            )
            return Series(other)
        #   ==================================
        for col in unsupported_cols:
            series = df[col]
            try:
                ClusterTable.fill_series(series)
            except Exception as e:
                series = df[col] = ''
        col_other = create_col_other()
        return col_other

    @staticmethod
    def unserialize_misc_cols(df):
        """
        Convert json-serialized misc column into its component columns.
        """
        import json
        #   Unserialize json columns
        if 'misc' in df.columns:
            misc_col = df.misc
            misc_col = misc_col.map(json.loads)
            values_misc = list(misc_col.values)
            if values_misc:
                df_misc = DataFrame(values_misc, index=misc_col.index)
                df = pd.concat([df, df_misc], axis=1)
            df = df.drop(['misc'], axis=1)
        return df

    def open(self, callback, **callback_kwargs):
        #   ------------------------------------------------------------
        LOGGER.info('\nWriting data to table {0} ({1} old data)...\n'.format(
            colored(self.h5fname, 'red'),
            colored('Appending to', 'yellow') if self.h5fmode == 'a' else colored('Removing', 'yellow')
        ))
        #   ------------------------------------------------------------
        os_path = os.path.join(self.base_path, self.h5fname)
        with open_file(os_path, mode=self.h5fmode, title=title) as h5file:
            return callback(h5file=h5file, **callback_kwargs)

    def create_groups(self):
        #   ------------------------------------------------------------
        LOGGER.info('\nWriting data to table {0} ({1} old data)...\n'.format(
            colored(self.h5fname, 'red'),
            colored('Appending to', 'yellow') if self.h5fmode == 'a' else colored('Removing', 'yellow')
        ))
        #   ------------------------------------------------------------
        os_path = os.path.join(self.base_path, self.h5fname)
        title = "Clusters"
        with open_file(os_path, mode=self.h5fmode, title=title) as h5file:
            #   Group containing individual clusters below.
            group_name = '_'.join([self.name[:4], 'cls'])
            human_group_name = '{0} Clusters'.format(self.name)
            group = self.get_or_create_group(
                self.parent_path, group_name, human_group_name , h5file
            )
            group_path = group._v_pathname
            parent = h5file.get_node(self.parent_path)
            tags = getattr(parent.attrs, 'label', '') if hasattr(parent, 'attrs') else []
            #   Group data by cluster id, and put into separate tables within the parent group.
            cluster_ids = self.cluster_ids
            for cluster_id in cluster_ids:
                cl_sample, cl_group_path, cl_table_name,  cl_size = self.create_cluster(
                    cluster_id, path=group_path, tags=tags, h5file=h5file,
                )
                if self.info_feats is not None:
                    try:
                        info_feats = self.info_feats.ix[cluster_id]
                    except KeyError as e:
                        info_feats = []
                else:
                    info_feats = []

                ##########
                LOGGER.info('Informative features: {0}'.format(
                    colored(pformat(info_feats), 'magenta')
                ))
                # pdb.set_trace()
                ##########

                #   Return an identifier to the current cluster group and a sample of the data
                yield cl_sample, cl_group_path, cl_table_name, cl_size, tags, info_feats
            #   Delete any table instances hanging off the path passed into this function
            #   (Their content is contained in the clusters we return.)
            self.delete_parent(parent, h5file)

    def create_cluster(self, cluster_id, path, tags, h5file):
        #   ------------------------------------------------------------
        LOGGER.info('\n\tPersisting data for cluster {0}...\n'.format(colored(cluster_id, 'red')))
        #   ------------------------------------------------------------
        df = self.df
        group_name = '{0}_dt'.format(cluster_id[:4])
        cluster_group = self.get_or_create_group(
            parent_path=path,
            name=group_name,
            human_name='{0} Data'.format(cluster_id),
            h5file=h5file
        )
        #   Filter the current df to include only the current cluster
        cluster_df = df[df.cluster_id == cluster_id]
        #   Drop the current cluster from the overall df to reduce filter time
        #   next iteration.
        df = df[~df.index.isin(cluster_df.index)]
          #   Make sure all document ids in the cluster df are unique.
        try:
            assert len(cluster_df) == len(cluster_df.id.unique())
        except AssertionError as e:
            LOGGER.error('Warning: {0}'.format(colored('IDs are non-unique', 'red')))
        cluster_table = self.get_or_create_table(
            cluster_group,
            name=cluster_id,
            human_name="{0} Table".format(cluster_id),
            h5file=h5file
        )
        #   Update table tags
        self.label(cluster_table, tags)
        total = len(cluster_df)
        # existing_ids = self.update_existing(cluster_df, cluster_table)
        #   Remove existing rows from the data.
        #####
        #   TODO:   Should the following line be this?
        #               cluster_df = cluster_df[~cluster_df.id.isin(existing_ids)]
        # curr_df = cluster_df[~cluster_df.id.isin(existing_ids)]
        #####
        new_ids = self.insert_new(cluster_df, cluster_table)
        ids = set(cluster_table.col('id'))
        cluster_df = cluster_df[cluster_df.id.isin(new_ids)]
        cluster_df['row_idx'] = xrange(len(cluster_df))
        cluster_table_name = cluster_table.name
        cluster_group_path = cluster_group._v_pathname
        #   -----------------------------------------
        LOGGER.info('\t{0} unique items are in {1}\n'.format(
            colored(str(len(ids)), 'cyan'),
            colored(cluster_table_name, 'red')
        ))
        ###############
        # pdb.set_trace()
        # LOGGER.debug(self.table_to_df(cluster_table))
        ###############
        #   -----------------------------------------
        sample = ClusterDataFrame.sample(cluster_df)
        size = len(cluster_df)
        return sample, cluster_group_path, cluster_table_name, size

    def delete_parent(self, parent, h5file):
        #   Delete any table instances hanging off the path passed into this function
        #   (Their content is contained in the clusters we return.)
        for leaf_name in list(parent._v_leaves):
            leaf_path = '/'.join([parent._v_pathname, leaf_name])
            #   ------------------------------------------------------------
            LOGGER.info('\n\tDeleting leaf {0}...\n'.format(colored(leaf_name, 'red')))
            #   ------------------------------------------------------------
            try:
                h5file.remove_node(leaf_path)
            except Exception as e:
                LOGGER.error(e)
                # pdb.set_trace()
            else:
                #   --------------
                LOGGER.info('\tFinished!')
                #   --------------

    def get_or_create_group(self, parent_path, name, human_name, h5file):
        path = '/'.join([parent_path, name])
        if path not in h5file:
            group = h5file.create_group(parent_path, name, human_name)
        else:
            group = h5file.get_node(path)
        return group

    def get_or_create_table(self, group, name, human_name, h5file):
        schema = self.schema
        if name not in group:
            #   Create a new table `name` under the group `group`.
            table = h5file.create_table(
                group, name, schema, human_name
            )
            cluster_id = group._v_pathname.rsplit('/', 2)[-2] + '_' + name
            #########
            LOGGER.debug("{} {} ----> {}".format(name, group._v_pathname, cluster_id))
            #########
            table._v_attrs.cluster_id = cluster_id
        else:
            table = getattr(group, name)
        return table

    def update_existing(self, df, table):
        total = len(df)
        existing_rows = [ r for r in table.iterrows() if r['id'] in df.id ]
        existing_ids = []
        num_updated = 0
        for row in existing_rows:
            #   Update existing rows.
            df_row = df.ix[row['id']]
            clean_content = df_row['clean_content']
            clean_tokens = df_row['clean_tokens']
            existing_ids.append(row['id'])
            row.update()
            num_updated += 1
        #   Flush the table's I/O buffer to persist to disk.
        table.flush()
        #   ------------------------------------------------
        LOGGER.info('\n\tUpdated {0} / {1} documents in {2}\n'.format(
            colored(num_updated, 'cyan'),
            colored(total, 'cyan'),
            colored(table.name, 'red')
        ))
        #   ------------------------------------------------
        return existing_ids

    def insert_new(self, df, table):
        total = len(df)
        inserted_ids = []
        num_inserted = 0
        table_row = table.row
        for ctr, (idx, df_row) in enumerate(df.iterrows()):
            #   Add new rows.
            self.insert_table_row(df_row, table_row, table)
            inserted_ids.append(df_row['id'])
            num_inserted += 1
        #   Flush the table's I/O buffer and persist to disk.
        table.flush()
        #   ------------------------------------------------
        LOGGER.info('\n\tInserted {0} / {1} documents into {2}\n'.format(
            colored(num_inserted, 'cyan'),
            colored(total, 'cyan'),
            colored(table.name, 'red')
        ))
        #   ------------------------------------------------
        return inserted_ids

    def insert_table_row(self, df_row, table_row, table):
        _id = df_row['id']
        clean_content = df_row['clean_content']
        clean_tokens = df_row['clean_tokens']
        misc = df_row['misc']
        #   Insert a new doc.
        table_row['id'] = _id
        table_row['clean_content'] = clean_content
        table_row['clean_tokens'] = clean_tokens
        table_row['misc'] = misc
        table_row.append()

    @classmethod
    def remove_table_row(cls, table, idx=None):
        ##########
        # LOGGER.debug(table.read(idx, idx + 1))
        ##########
        #   TODO:   In the pytables 3.0 API, an stop index must be supplied.
        table.remove_rows(idx)

    @classmethod
    def update_labels(cls, labels, h5fname, group_path, table_name, **kwargs):
        #   Load and then label via a callback to the load function.
        return cls.load(
            h5fname, group_path, table_name,
            callback=cls.label, values=labels,
            **kwargs
        )

    @classmethod
    def load(cls, h5fname, group_path, table_name, callback=None, **callback_kwargs):
        #   Load the cluster data
        os_path = os.path.join(cls.base_path, h5fname)
        with open_file(os_path, mode="r+") as h5file:
            path = '/'.join([group_path, table_name])
            try:
                node = h5file.get_node(path)
            except Exception as e:
                LOGGER.error(colored(e, 'red'))
                LOGGER.warn('Looking for a group...')
                ###############
                # pdb.set_trace()
                ###############
                try:
                    #   Try to retrieve the group
                    node = h5file.get_node(group_path)
                except Exception as e:
                    LOGGER.error(colored(e, 'red'))
                    ###############
                    # pdb.set_trace()
                    ###############
            if callback is not None:
                return callback(node, **callback_kwargs)

    def postprocess(fname='pm_cluster.csv'):
        collection = self.mongo_session.collection
        path = self.data_path
        #   Get the clustered data
        #   Use 'id' as the index.
        df = pd.read_csv(path)
        df = df.set_index('id')
        #   Convert comma-separated label strings to lists.
        df.label = df.label.fillna('')
        if any(df.label.map(lambda x: isinstance(x, basestring))):
            df.label = df.label.map(lambda x: x.split(','))
            df.label = df.label.map(lambda l: set([t.lower().strip() for t in l]))
        new_rows = []
        for idx, (k, row) in enumerate(df.iterrows()):
    #         if idx > 100: break
            db_post = collection.find_one({'_id': k})
            d = self.row_to_dict(row, db_post)
            #   Update the post with new tags
            labels = [ l for l in list(d['label']) if l ]
            if labels: self.update_labels(k, labels)
            for rd in self.ravel_label_row(d):
                rd.update(d)
                new_rows.append(rd)
        new_df = DataFrame(new_rows)
        if to_csv: self.to_csv(new_df, self.final_base_path, index=False)
        return new_df

    @classmethod
    def export(
            cls, h5fname, group_path, fname=None,
            db_name=None, db_type=None,
            mongo_session_maker=None,
        ):
        ########
        #   DEBUGGING
        # db_name = 'pm_web_comments'
        # pdb.set_trace()
        ########
        fname = fname or '/'.join([h5fname[:-3], group_path])
        if not fname.endswith('.csv'): fname += '.csv'
        os_path = os.path.join(cls.base_path, h5fname)
        out_fname = fname.strip('/').replace('/', '_')[:-4] + '_labeled.csv'
        out_path = os.path.join(cls.base_path, 'labeled', out_fname)
        with open_file(os_path, mode="r+") as h5file:
            ########
            # pdb.set_trace()
            ########
            table_path = group_path
            table_seq = h5file.walk_nodes(table_path, classname='Table')
            #   Convert each table to a dataframe, adding any attached labels.
            dfs = [ cls.table_to_df(t, add_labels=True) for t in table_seq ]
            ########
            # pdb.set_trace()
            ########
            #   TBD: Not here...
            # mongo_save = False
            # if db_name:
            #     try:
            #         mongo_session = MongoSession(db_name=db_name)
            #         collection = mongo_session.collection
            #     except Exception as e:
            #         LOGGER.error(colored(e ,'red'))
            #     else:
            #         mongo_save = True
            new_rows = []
            for df in dfs:
                for idx, (k, row) in enumerate(df.iterrows()):
                    d = row.to_dict()
                    ##########
                    LOGGER.debug(colored(pformat(d), 'magenta'))
                    # pdb.set_trace()
                    ##########
                    #   Update the post with new tags
                    labels = [
                        l for l in list(d.pop('labels', '').split('|'))
                            if len(l) > 1
                    ]
                    ##########
                    LOGGER.debug(colored(pformat(labels), 'magenta'))
                    ##########
                    #   TBD: Not here...
                    # if mongo_save:
                        # d = ClusterTable.to_mongo(d, session)
                    if labels:
                        for rd in ClusterDataFrame.ravel_label_row(d):
                            rd.update(d)
                            new_rows.append(rd)
                    else:
                        d['label'] = ''
                        new_rows.append(d)
            new_df = DataFrame(new_rows)
            ############
            # LOGGER.debug(df)
            LOGGER.debug(colored(new_df, 'magenta'))
            # pdb.set_trace()
            ############
            df = new_df
            df.to_csv(out_path, index=False)
        return df, out_fname

    @classmethod
    def to_mongo(doc, session):
        _id = doc.get('id', None)
        if _id is not None:
            #   Add some other columns from the database.
            db_post = session.collection.find_one({'_id': _id})
            if db_post is not None:
                doc = session.row_to_dict(doc, db_post)
                doc['labels'] = list(set(labels) | set(db_post.get('labels', [])))
                doc['labels'] = [ l for l in doc['labels'] if len(l) > 1 ]
                session.update_labels(_id, labels)
        return doc


    @classmethod
    def load_and_remove_from_id(cls, _id):
        h5fname, group_path, table_name = cls.parse_id(_id)
        return cls.load(h5fname, group_path, table_name, callback=cls.remove)

    @classmethod
    def load_and_remove(cls, h5fname, group_path, table_name):
        return cls.load(h5fname, group_path, table_name, callback=cls.remove)

    @staticmethod
    def label(table, values=None, db_name=None, db_type=None, **kwargs):
        if values is None:                      values = []
        elif not hasattr(values, '__iter__'):   values = [values]
        values = [ v.strip().lower() for v in values ]

        def get_labels(node, new_labels=None):
            try:
                labels = node._v_attrs.labels
            except AttributeError:
                labels = sorted(list(set(new_labels))) if new_labels else []
            else:
                if new_labels:
                    labels = sorted(list(set(labels) | set(new_labels)))
            labels = [ l for l in labels if l ]
            return labels

        def save_to_mongo(table, labels, db_name):
            try:
                mongo_session = MongoSession(db_name=db_name)
                collection = mongo_session.collection
            except Exception as e:
                LOGGER.error(colored(e ,'red'))
            else:
                id_array = table.col('id')
                id_array = list(id_array)
                while id_array:
                    ids = id_array[:1000]
                    collection.update(
                        {'_id': {'$in': ids}},
                        {
                            '$addToSet': {
                                'labels': {
                                    '$each': labels
                                }
                            }
                        }
                    )
                    offset = min(len(id_array), 1000)
                    id_array = id_array[offset:]
                is_save = True

        def save_to_elasticsearch(table, labels, db_name):
            id_array = table.col('id')
            #   TBD
            with open("labels_temp_{}.csv".format(datetime.datetime.now().strftime('%Y%m%dT%H%M%S'))) as f:
                writer = csv.DictWriter(f, fieldnames=['_id', 'labels'])
                for _id in id_array:
                    writer.writerow(
                        {'_id': _id, 'labels': u'|'.join(labels) }
                    )
            # "script": "if (ctx._source.containsKey(\"paymentInfos\")) {ctx._source.paymentInfos += paymentInfo;} else {ctx._source.paymentInfos = [paymentInfo]}"
            # id_array = list(id_array)
            # docs = (
            #     {
            #         '_id': _id, 'doc': {

            #         }
            #     },
            #         for _id in id_array
            # )
            # try:
            #     eshelpers.to_elasticsearch(
            #         grp, brand, 'promotion', action='update', **kwargs
            #     )
            # except Exception as e:
            #     LOGGER.error(colored(e ,'red'))
            # else:

        def save_to_db(table, labels, db_name=None, db_type=None):
            ###############
            # pdb.set_trace()
            ###############
            is_save = False
            # if db_name:
            #     if db_type == 'mongo':
            #         save_to_mongo(table, labels, db_name)
            #     elif db_type == 'elasticsearch':
            #         save_to_elasticsearch(table, labels, db_name)
            id_array = table.col('id')
            #   TBD
            with open("labels_temp_{}.csv".format(datetime.datetime.now().strftime('%Y%m%dT%H%M%S')), 'wb') as f:
                fieldnames = ['_id', 'labels']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(dict(zip(fieldnames, fieldnames)))
                for _id in id_array:
                    writer.writerow(
                        {'_id': _id, 'labels': u'|'.join(labels) }
                    )
        result = {}
        if values:
            #   Get previously applied labels.
            prev_labels = get_labels(table)
            #   Get the parent.
            parent = table._v_parent
            ##########
            # LOGGER.debug(prev_labels)
            # pdb.set_trace()
            ##########
            if prev_labels:
                #   We've already updated labels for this table.
                #   Overwrite the old with the new.
                labels = sorted(list(set(values)))
                labels = [ l for l in labels if l ]
                ##########
                # LOGGER.debug(labels)
                # pdb.set_trace()
                ##########
            else:
                #   No labels have been applied. Add the parent labels to the
                #   supplied values.
                labels = get_labels(parent, values)
            #   TODO:   Decide whether to do this here or in `export`.
            save_to_db(table, labels, db_name=db_name, db_type=None)
            #   Update labels for the parent group.
            parent._v_attrs.labels = labels
            labels_str = colored(', '.join(labels), 'yellow')
            #   --------------------------------------------
            LOGGER.info('Added labels {0} to Parent Group {1}\n'.format(
                labels_str,
                colored(parent._v_pathname, 'cyan'),
            ))
            #   ---------------------------------------------
            #########
            # LOGGER.debug(labels)
            # LOGGER.debug(parent._v_attrs.labels)
            # pdb.set_trace()
            #########
            #   Recursively add labels to children of the parent group (including
            #   the current table).
            children = parent._f_walknodes()
            child_paths = []
            for child in children:
                #   Update labels for the current table and its children.
                if child == table:
                    child_labels = labels
                else:
                    child_labels = get_labels(child, labels)
                #   TODO:   Decide whether to do this here or in `export`.
                # save_to_db(child, child_labels, db_name)
                child._v_attrs.labels = child_labels
                #########
                # LOGGER.debug(child)
                # LOGGER.debug(child_labels)
                # pdb.set_trace()
                #########
                child_pathname, child_name = child._v_pathname, child._v_name
                child_path = '/'.join((child_pathname, child_name))
                child_paths.append(child_pathname)
                result[child_pathname] = child_labels
            #   --------------------------------------------
            LOGGER.info('Added labels {0} to Children\n{1}\n\n'.format(
                labels_str,
                colored(
                    '\n'.join(['\t{0}'.format(p) for p in child_paths]),
                    'cyan'
                )
            ))
            #   ---------------------------------------------
        elif not hasattr(table.attrs, 'labels'):
            table.attrs.labels = []
        return result

    @staticmethod
    def update_info_feats(table, values):
        table._v_attrs.info_feats = values

    @staticmethod
    def remove(node):
        #   -------------------------------------------------------------
        LOGGER.info(colored('\tRemoving node {0}\n'.format(node._v_name), 'red'))
        #   -------------------------------------------------------------
        # node.remove()
        ######
        # LOGGER.debug(node)
        # LOGGER.debug(type(node))
        # pdb.set_trace()
        ######
        node._f_remove(recursive=True, force=True)

    @staticmethod
    def table_to_df(table, add_labels=False):
        data = [ dict(zip(table.cols._v_colnames, t)) for t in table.read()]
        df = DataFrame(data)
        df['cluster_id'] = getattr(table.attrs, 'cluster_id')
        if add_labels:
            #   Add table labels
            labels = getattr(table.attrs, 'labels', [])
            labels = '|'.join(labels)
            df['labels'] = labels
        df = ClusterTable.unserialize_misc_cols(df)
        return df

    @staticmethod
    def sample(table):
        df = ClusterTable.table_to_df(table)
        samples = ClusterDataFrame.sample(df)
        return samples

    @staticmethod
    def create_id(h5fname, path, name):
        return '-'.join([
            h5fname.replace('/', '-'),
            path.replace('/', '-'),
            name.replace('/', '-')
        ])

    @staticmethod
    def parse_id(_id):
        _id = _id.replace('-', '/')
        h5fname, path = _id.split('/', 1)
        group_path, table_name = path.rsplit('/', 1)
        return h5fname, group_path, table_name



