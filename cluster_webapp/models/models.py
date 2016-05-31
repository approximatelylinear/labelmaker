
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
from models_base import (
    ClusterDataFrame, vec_maybe_date_parse, vec_maybe_join_seq,
    vec_maybe_join_seq_comma, vec_maybe_encode_utf8, Cluster, ClusterTable
)
from models_pm import PMClusterDataFrame
from models_es import ESClusterDataFrame
from models_topsy import TopsyClusterDataFrame
from models_boardreader import BRClusterDataFrame

from lib.clustering import clusterers
from lib.text_processing import clean as text_clean

#: The directory containing this script
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

#: Logger name
LOGGER = logging.getLogger("base.models")

#: Elasticsearch logger
LOGGER_ES = logging.getLogger('elasticsearch')
LOGGER_ES_TR = logging.getLogger('elasticsearch.trace')


#   Utilities
def has_iter(x): return hasattr(x, '__iter__')
vec_has_iter = np.vectorize(has_iter)



from sklearn.svm import LinearSVC
class L1LinearSVC(LinearSVC):
    """
    Linear SVM with l1 penalty.
    """
    def fit(self, X, y):
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        self.transformer_ = LinearSVC(
            penalty="l1", dual=False, tol=1e-3
        )
        X = self.transformer_.fit_transform(X, y)
        return LinearSVC.fit(self, X, y)

    def predict(self, X):
        X = self.transformer_.transform(X)
        return LinearSVC.predict(self, X)


class Classifier(object):
    from sklearn import cross_validation
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.linear_model import RidgeClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.linear_model import Perceptron
    from sklearn.linear_model import PassiveAggressiveClassifier
    from sklearn.naive_bayes import BernoulliNB, MultinomialNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neighbors import NearestCentroid
    from sklearn.utils.extmath import density
    from sklearn import metrics

    def __init__(self, df_wrapper, labels=None):
        self.df_wrapper = df_wrapper
        self.labels = labels
        self.collection = self.df_wrapper.mongo_session.collection

    def run(self):
        labels, df = self.get_training_data()
        dfs, unlabel_posts = self.get_data()
        self.classify(dfs, unlabel_posts)

    def get_training_data(self):
        #   Find posts with labels
        query = {}
        if self.labels:
            query['labels'] = {'$in': self.labels}
        else:
            query['labels'] = {'$not': {'$size': 0}}
        label_posts = self.collection.find(query)
        target = []
        data = []
        skip_labels = self.skip_labels
    #     for idx, (k, row) in enumerate(df.iterrows()):
        for idx, d in enumerate(label_posts):
            #################
            if idx % 1000 == 0:
                LOGGER.info('[{0}] {1}\n'.format(colored(str(idx), 'red'), pformat(d)))
            #################
            for rd in ravel_label_row(d):
                rd.update(d)
                if not (self.labels and not rd['label'] in self.labels):
                    target.append(rd['label'])
                    data.append(rd['clean_content'])
        labels = list(set(target))
        code_map = dict( zip(labels, range(len(labels))) )
        target_codes = [ code_map[s] for s in target ]
        target_labels = target
        target = target_codes
        #   ==================================================
        LOGGER.debug(pformat(zip(target_labels, target)[:20]))
        #   ==================================================
        df = DataFrame({'target': target_labels, 'data': data})
        return labels, df

    def get_data(self, labels, df):
        dfs = {}
        for label in labels:
            curr_df = df.copy()
            curr_df.target = np.where(df['target'] == label, 1, 0)
            #   Remove duplicate combinations of target and data.
            curr_df = curr_df.drop_duplicates(['target', 'data'])
            dfs[label] = curr_df
        unlabel_posts = collection.find({'labels': {'$size': 0}}) #, limit=100)
        unlabel_posts = [ {'_id': d['_id'], 'clean_content': d['clean_content']} for d in unlabel_posts ]
        return dfs, unlabel_posts

    @staticmethod
    def fit(X, y, X_pred):
        vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            max_df=0.5,
            stop_words='english'
        )
        X = vectorizer.fit_transform(X)
        X_pred = vectorizer.transform(X_pred)
        feature_names = np.asarray(vectorizer.get_feature_names())
        n_samples, n_feats = X.shape
        LOGGER.info("Train n_samples: %d, n_features: %d" % X.shape)
        LOGGER.info("Extracting 1000 best features by a chi-squared test")
        ch2 = SelectKBest(chi2, k=min(n_feats, 1000))
        X = ch2.fit_transform(X, y)
        X_pred = ch2.transform(X_pred)
        # clf = L1LinearSVC()
        clf = MultinomialNB(alpha=.01)
        clf.fit(X, y)
        pred = clf.predict(X_pred)
        return pred

    def classify(self, dfs, unlabel_posts):
        for label, df in dfs.iteritems():
            ######
            LOGGER.info('Classifying docs with {0}'.format(colored(label, 'red')))
            LOGGER.info(df.data[:5])
            LOGGER.info(df.target[:5])
            ######
            unlabel_data = [ d['clean_content'] for d in unlabel_posts ]
            pred = self.fit(
                np.array(df.data),
                np.array(df.target),
                unlabel_data
            )
            pred_df = DataFrame(
                {
                    'label': pred,
                    'clean_content': unlabel_data,
                    '_id': [d['_id'] for d in unlabel_posts]
                }
            )
            pred_df = pred_df[pred_df.label > 0]
            LOGGER.info('{0} items labeled {1}'.format(colored(str(len(pred_df)), 'red'), colored(label, 'yellow')))
            for idx, (_, row) in enumerate(pred_df.iterrows()):
                d = row.to_dict()
                # LOGGER.debug(pformat(d))
                # pdb.set_trace()
                #################
                if idx % 1000 == 0:
                    LOGGER.info('[{0}] {1}\n'.format(colored(str(idx), 'red'), pformat(d)))
                #################
                self.collection.update(
                    {'_id': d['_id']},
                    {
                        '$set'      : {'train': False},
                        '$addToSet' : {'labels': label}
                    }
                )

    def export(self):
        return self.df_wrapper.export()












