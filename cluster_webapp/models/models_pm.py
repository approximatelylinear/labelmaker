

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

#   Custom
from models_mongo import MongoClusterDataFrame, MongoSession
import models_base
import pm_comment_models
# from pm_comment_models import connect as pm_comment_connect


#: The directory containing this script
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

#: Logger name
LOGGER = logging.getLogger("base.models.pm")


class PMClusterDataFrame(MongoClusterDataFrame):
    #: Base directory of metadata.
    meta_base_path = os.path.join(THIS_DIR, 'data', 'meta', 'pm_comments')
    if not os.path.exists(meta_base_path): os.makedirs(meta_base_path)
    #: Base directory of labels.
    labels_path = os.path.join(meta_base_path, 'pm_labels.csv')
    #: Columns with text
    text_cols = [
        'clean_content',
        'conversation',
        'clean_tokens',
        'content',
    ]
    #: Columns to drop when creating the dataframe.
    drop_cols = [
        '_cls',
        'category',
        'channel',
        'eventgroup_id',
        'eventhistory_id',
        'media_id',
        'winning_ccn',
        'contact_date',
    ]
    #: Columns to permit users to filter.
    info_cols = [
        'atc_id',
        'clean_content',
        'conversation',
        'section',
        'date',
        'labels',
        'media_url',
        'topics',
    ]
    info_fields = [ {'name': col} for col in info_cols ]
    fld_date = 'date'
    #: All relevant event ids
    event_ids = [
        'EVTUGCLIKEEXTERNALME',
        'EVTUGCLIKEPHOTO',
        'EVTUGCLIKEEXTERNALMEDIA',
        'EVTUGCPOSTPHOTOCOMME',
        'EVTUGCPOSTPHOTOCOMMENT',
        'EVTUGCPOSTEXTERNALME',
        'EVTUGCPOSTEXTERNALMEDIACOMMENT'
    ]
    #: Event ids indicating comments
    eventids_comment = [
        'EVTUGCPOSTEXTERNALME',
        'EVTUGCPOSTPHOTOCOMME',
        'EVTUGCPOSTPHOTOCOMMENT',
        'EVTUGCPOSTEXTERNALMEDIACOMMENT'
    ]
    #: Event ids indicating likes
    eventids_like = [
        'EVTUGCLIKEEXTERNALME',
        'EVTUGCLIKEEXTERNALMEDIA',
        'EVTUGCLIKEPHOTO',
    ]

    def __init__(
            self,
            db_name='pm_web_comments',
            refresh=False,
            **kwargs
        ):
        self.fname = '{0}.csv'.format(db_name)
        super(PMClusterDataFrame, self).__init__(self.fname, **kwargs)
        self.db_name = db_name
        self.filters = kwargs.pop('filters', {})
        self.fields = kwargs.pop('fields', [])
        self.limit = self.filters.pop('limit', 100)
        self.is_labeled = self.filters.pop('is_labeled', None)

        self.base_query = {
            '_cls': 'Post',
        }
        #   Initialize the Mongo connections
        self.mongo_session = PMMongoCluster(db_name=self.db_name, base_query=self.base_query)

    def get_info(self):
        filters = [
            self.get_choices('labels'),
            self.get_choices('conversation'),
            self.get_choices('section'),
            self.get_choices('media_url'),
            self.get_choices('topics'),
            { 'name': 'event_id', 'choices': ['comment', 'like', 'all']},
        ]
        # count = self.mongo_session.collection.count()
        count = 'A Lot'
        result = {
            'fields'    : self.info_fields,
            'filters'   : filters,
            'count'     : count,
        }
        return result

    def scratch_20130723(self):
        #   Find posts with the 'food' topic
        query = copy.deepcopy(self.base_query)
        query['topics'] = 'food'
        query['label_count'] = {'$gt': 0}
        # query['labels'] = {'$not': {'$size': 0}}
        # params = {'limit': 2000}
        params = {}
        collection = self.mongo_session.collection
        posts = collection.find(query, **params)
        ids = []
        for post in posts:
            ##########
            LOGGER.debug(colored(pformat(post), 'magenta'))
            ##########
            if 'labels' not in post:
                continue
            # labels = post['labels']
            _id = post['_id']
            ids.append(_id)
            ng_labels = [
                'food',     #   Remove the 'food' label
                '__none__', #   Remove the placeholder
            ]
            #   Remove labels that are a single letter
            char_labels = [ l for l in post['labels'] if len(l) <= 1 ]
            ng_labels.extend(char_labels)
            collection.update(
                {'_id': _id},
                {
                    '$pullAll': {
                        'labels': ng_labels
                    },
                }
            )
        new_posts = collection.find({'_id': {'$in': ids[:1000]}})
        LOGGER.debug(colored(pformat(list(new_posts)), 'cyan'))

    def export(self, to_csv=True):
        #   Return all posts in csv format
        collection = self.mongo_session.collection
        #   Get the clustered and labeled data
        #   Find posts with the 'food' topic
        query = copy.deepcopy(self.base_query)
        query['topics'] = self.topic
        query['label_count'] = {'$gt': 0}
        # query['labels'] = {'$not': {'$size': 0}}
        # params = {'limit': 10}
        params = {}
        posts = collection.find(query, **params)
        ids = []
        new_rows = []
        for post in posts:
            if 'labels' not in post:
                continue
            d = post
            d.pop('_cls', None)
            #   Update the post with new tags
            for rd in self.ravel_label_row(d):
                rd.update(d)
                new_rows.append(rd)
        new_df = DataFrame(new_rows)
        df = new_df
        df.label = df.label.fillna('')
        #   Convert comma-separated label strings to lists.
        if any(df.label.map(lambda x: hasattr(x, '__iter__'))):
            df.label = df.label.map(lambda l: set([t.lower().strip() for t in l]))
            df.label = df.label.map(lambda x: '|'.join(x))
        if any(df.topics.map(lambda x: hasattr(x, '__iter__'))):
            df.topics = df.topics.map(lambda l: set([t.lower().strip() for t in l]))
            df.topics = df.topics.map(lambda x: '|'.join(x))
        if to_csv:
            fname = self.db_name + '_' + self.topic + '.csv'
            path = os.path.join(self.final_base_path, fname)
            self.to_csv(df, path, index=False)
        return df

    def row_to_dict(self, row, db_post):
        """
        Converts a dataframe row to a dictionary.

        """
        d = row.to_dict()
        d['id'] = k
        d['media_url'] = db_post['media_url']
        d['conversation'] = db_post['conversation']
        d['atc_id'] = db_post['atc_id']
        d['date'] = db_post['date']
        return d

    def find_posts(self, filters=None):
        """
        Finds and returns posts in mongo matching specified filters.

            1. Gets and formats filters
            2. Calls mongo_session.find_posts with specified filters

        :return: DataFrame with relevant data returned from Mongo.
        :rtype: pd.DataFrame

        """
        if filters is None: filters = self.get_filters()
        params = {}
        limit = self.limit
        if limit is not None: params['limit'] = limit
        query = copy.deepcopy(self.base_query)
        #   Add event ids
        event_ids = filters.pop('event_ids', 'comment')
        query.update(self.get_event_id_filter(event_ids))
        #   Add the media url.
        media_url = filters.pop('media_url', None)
        if media_url is not None:
            query.update(self.get_media_url_filter(media_url))
        #   Get items that have / haven't been labeled yet.
        is_labeled = self.is_labeled
        query.update(self.get_is_labeled_filter(is_labeled))
        #   Update the query with user-selected filters that don't require special
        #   formatting.
        query.update(filters)
        #   --------------------------------------------------------------
        LOGGER.info('Looking for {0} posts matching {1}'.format(
            colored(limit or 'all', 'red'),
            colored(pformat(query), 'magenta')
        ))
        #   --------------------------------------------------------------
        curr_df = self.mongo_session.find_posts(query, **params)
        return curr_df

    def get_media_url_filter(self, media_url):
        query = {}
        if hasattr(media_url, '__iter__'):
            query['media_url'] = {'$in': media_url}
        else:
            #   Treat the media url as a regular expression.
            # query['media_url'] = "/^{0}/".format(media_url)
            query['media_url'] = media_url
        return query

    def get_event_id_filter(self, event_ids):
        query = {}
        if (event_ids is None) or (event_ids == 'all'):
            event_ids = self.event_ids
        elif event_ids == 'comment':
            event_ids = self.eventids_comment
        elif event_ids == 'like':
            event_ids = self.eventids_like
        query['event_id'] = {'$in': event_ids}
        return query

    def get_filters(self):
        """
        Get filters

        """
        query = {}
        filters = copy.deepcopy(self.filters)
        #   Remove non-query-related filters
        filters.pop('limit', None)
        filters.pop('is_labeled', None)
        for k, v in filters.iteritems():
            if v:
                if k in ['date']:
                    query['date'] = date_parse(v)
                else:
                    query[k] = v
        return query

    def get_topics(self):
        """
        Load topics from :py:attr:`.labels_path` and return them as a DataFrame.

        """
        with open(self.labels_path, 'rbU') as f:
            reader = csv.DictReader(f)
            topic_data = list(reader)
        topic_df = DataFrame(topic_data)
        topic_df = topic_df.set_index('media_url')
        return topic_df

    def get_media_urls_for_topic(self, topic, topic_df):
        return set(topic_df[topic_df['type'] == topic].index)

    def postprocess(fname='pm_cluster.csv'):
        pass


class PMMongoSession(MongoSession):
    def __init__(self, *args, **kwargs):
        super(PMMongoSession, self).__init__(*args, **kwargs)
        # pm_comment_connect(self.db_name)
        # self.db = pm_comment_models.Post._get_db()
        # self.collection = pm_comment_models.Post._get_collection()

    @staticmethod
    def row_to_dict(d, db_post):
        #   d = row.to_dict()
        #   Format the dates
        d['time'] = db_post['date'].strftime('%H:%M:%S')
        d['date'] = db_post['date'].strftime('%m/%d/%Y')
        d['conversation'] = db_post['conversation'].encode('utf-8')
        d['atc_id'] = db_post['atc_id']
        d['media_url'] = db_post['media_url'].encode('utf-8')
        return d
