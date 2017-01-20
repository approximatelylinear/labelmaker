
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
from termcolor import colored, cprint
from dateutil.parser import parse as date_parse
from pymongo import MongoClient

#   Custom
from cluster_webapp.models_base import ClusterDataFrame

#: The directory containing this script
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

#: Logger name
LOGGER = logging.getLogger("base.models.mongo")



class MongoClusterDataFrame(ClusterDataFrame):
    """
    :py:class:`ClusterDataFrame` sourced from Mongo

    """

    def get_choices(self, name, field=None):
        """
        Get values for specified field.

        :param str name: Name of field in output.
        :param str field: Name of field in input (defaults to name).

        """

        if field is None: field = name
        return {
            'name': name,
            'choices': list(self.mongo_session.collection.distinct(field))
        }

    def load(self, to_csv=True):
        """
        Loads data by calling:
            (1) :py:meth:`.load_from_mongo`
            (2) :py:meth:`._load_from_mongo`
            (3) :py:meth:`.find_posts`
            (4) :py:meth:`self.mongo_session.find_posts`
            (5) Sets :py:attr:`.df` to the loaded Dataframe.

        :param bool to_csv: Whether to save the data to csv.

        :return: Loaded dataframe
        :rtype: pd.DataFrame

        """
        df = self.load_from_mongo()
        self.df = df
        self.loaded = True
        return df

    def load_from_mongo(self, to_csv=True):
        """
        Loads data

            1. Calls :py:meth:`._load_from_mongo`
            2. Samples the data if the number of rows exceeds :py:attr:`.limit`
            3. Saves the data to csv if ``to_csv`` is True

        :param bool to_csv: Whether to save the data to a file in csv format.

        :return: Loaded dataframe
        :rtype: pd.DataFrame

        """
        df = self._load_from_mongo() or DataFrame()
        #   Sample the data down some more.
        if len(df) > self.limit:
            idxs = random.sample(df.index, self.limit)
            df = df.ix[idxs]
        if to_csv and not df.empty:
            self.to_csv(df, self.raw_path, index=False)
        return df

    def _load_from_mongo(self):
        """
        Loads data

            1. Calls :py:meth:`.find_posts`, which interacts with Mongo.
            2. Drops columns specified in :py:attr:`.drop_cols`
            3. Keeps only those columns specified in :py:attr:`.keep_cols`

        """
        curr_df = self.find_posts()
        drop_cols = [ col for col in self.drop_cols if col in curr_df ]
        keep_cols = [ c for c in self.keep_cols if c in curr_df ]
        if len(curr_df):
            curr_df.index = curr_df._id
            if drop_cols: curr_df = curr_df.drop(drop_cols, axis=1)
            #   General ??
            if keep_cols: curr_df = curr_df[keep_cols]
            curr_df = curr_df.fillna(0)
            #   General ??
            curr_df['label_choices'] = ''
            return curr_df

    def find_posts(self, filters=None):
        raise NotImplementedError


class MongoSession(object):
    """
    Wraps a connection to a MongoDB database collection named ``post``
    """

    def __init__(
            self,
            db_name=None,
            limit=100,
            refresh=False,
            base_query=None,
            **kwargs
        ):
        if base_query is None: self.base_query = {}
        #   Initialize the Mongo connections
        self.db_name = db_name
        connection = MongoClient()
        self.db = connection[db_name]
        self.collection = self.db.post

    def find_posts(self, query, **params):
        curr_posts = self.collection.find(query, **params)
        new_posts = []
        idx = None
        for idx, post in enumerate(curr_posts):
            if not post.get('labels'):
                post['labels'] = []
            new_posts.append(post)
        #   --------------------------------------------------------------
        LOGGER.info('Found {0} posts matching {1}'.format(
            colored(idx + 1 if idx is not None else 0, 'red'),
            colored(pformat(query), 'magenta')
        ))
        #   --------------------------------------------------------------
        curr_df = DataFrame(new_posts)
        return curr_df

    def update_labels(self, _id, labels):
        collection = self.collection
        #   ------------
        LOGGER.debug('MongoSession.update_labels: {}'.format(labels))
        #   ------------
        result = collection.find_and_modify(
            {'_id': _id},
            update={
                '$addToSet' : {'labels': {'$each': labels}},
            },
            # full_response=True,
            # new=True
        )
        result = collection.find_and_modify(
            {'_id': _id},
            update={
                '$pull'     : {'labels': '__none__'},    #   Remove the placeholder
            },
            full_response=True,
            new=True
        )
        #########
        # LOGGER.debug(pformat(result))
        # pdb.set_trace()
        #########
        if result['ok']:
            label_count = len(result['value']['labels'])
            result = collection.find_and_modify(
                {'_id': _id},
                update={
                    '$set'     : {'label_count': label_count}
                },
                new=True,
            )
            assert result['label_count'] == label_count
        # collection.update(
        #     {'_id': _id},
        #     {
        #         '$pull': {'labels': '__none__'},    #   Remove the placeholder
        #     }
        # )
        # collection.update(
        #     {'_id': _id},
        #     {
        #         '$addToSet' : {'labels': {'$each': labels}},
        #         '$inc'      : {'label_count': len(set(labels))}
        #     }

    @staticmethod
    def row_to_dict(d, db_post):
        return d


