

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
from dateutil.parser import parse as date_parse

#   Custom
from cluster_webapp.models_base import (
    ClusterDataFrame, vec_maybe_date_parse, vec_maybe_join_seq,
    vec_maybe_join_seq_comma, vec_maybe_encode_utf8
)

from cluster_webapp.lib.clustering import clusterers
from cluster_webapp.lib.text_processing import clean as text_clean

#: The directory containing this script
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

#: Logger name
LOGGER = logging.getLogger("base.models.es")

#: Elasticsearch logger
LOGGER_ES = logging.getLogger('elasticsearch')
LOGGER_ES_TR = logging.getLogger('elasticsearch.trace')


"""
Build query
-----------
    - Start with base query


filter on comment length

"""


class ESClusterDataFrame(ClusterDataFrame):
    #: Columns with text
    text_cols = [
        '_id',
        'id_user',
        'content',
        'site',
        'community',
    ]
    #: Columns with content to keep
    keep_cols = [
        '_id',
        'id_user',
        'content',
        'site',
        'status',
        'date_create',
        'community',
        'ct_words',
    ]
    drop_cols = []
    info_cols = keep_cols
    info_fields = [ {'name': col} for col in info_cols ]
    #   TBD adjust filter fields on a per-db basis.
    fltr_fields = [
        'labels', 'query', 'status', 'community', 'site', 'word_count'
    ]
    base_query = {'content': '*'}
    fld_date = 'date_create'

    def __init__(
            self,
            db_name=None,
            refresh=False,
            **kwargs
        ):
        self.fname = '{0}.csv'.format(db_name)
        super(ESClusterDataFrame, self).__init__(self.fname, **kwargs)
        self.db_name = db_name
        self.filters = kwargs.pop('filters', {})
        #   Restrict to text documents by default.
        self.filters.setdefault('type', 'text')
        self.filters.setdefault('randomize', True)
        self.fields = kwargs.pop('fields', [])
        #   Remove non-query-related filters
        self.limit = self.filters.pop('limit', 100)
        self.is_labeled = self.filters.pop('is_labeled', None)
        self.base_query = kwargs.get('base_query') or self.base_query
        #   TBD: Add doc type parameter.
        self.doc_type = kwargs.get('doc_type')
        #   Initialize the ES connections
        self.session = ElasticSearchSession(
            db_name=self.db_name,
            base_query=self.base_query, doc_type=self.doc_type
        )

    def fmt_choices(self, name, choices, field=None, count=None):
        """
        Get values for specified field.

        :param str name: Name of field in output.
        :param str field: Name of field in input (defaults to name).

        """
        if field is None: field = name
        #   ------
        def get_choices(choices):
            if isinstance(choices, dict):
                for k, v in choices.iteritems():
                    yield {'key': k, 'value': v}
            for c in choices:
                #   TBD: Make this applicable to cases where the value is other than 'doc_count'
                # key = c.pop('key')
                # if 'value' in c:
                #     value = c['value']
                # else:
                #     rmdr = c.keys()
                #     if rmdr:
                #         value = c[rmdr[0]]
                if 'doc_count' in c:
                    yield {'key': c['key'], 'value': c['doc_count']}
        #   ------
        return {
            'name': name,
            'choices': list(get_choices(choices))
        }

    def get_info(self):
        filter_counts = self.session.get_filter_counts(self.fltr_fields)
        count = filter_counts.pop('count')
        filters = [
            self.fmt_choices(k, v) for k, v in filter_counts.iteritems()
        ]
        result = {
            'fields'    : self.info_fields,
            'filters'   : filters,
            'count'     : count,
        }
        return result

    def load(self, to_csv=False):
        # if (not self.refresh) and (not os.path.exists(self.clean_path)):
        df = self._load()
        if df is None: df = DataFrame()
        if to_csv and not df.empty:
            self.to_csv(df, self.raw_path, index=False)
        self.df = df
        self.loaded = True
        return df

    def _load(self):
        drop_cols = self.drop_cols
        curr_df = self.find_posts()
        if len(curr_df):
            curr_df.index = curr_df._id
            drop_cols = [c for c in drop_cols if c in curr_df]
            curr_df = curr_df.drop(drop_cols, axis=1)
            keep_cols = [ c for c in self.keep_cols if c in curr_df ]
            curr_df = curr_df[keep_cols]
            curr_df['label_choices'] = ''
            return curr_df

    def find_posts(self, query=None, filters=None):
        params = {}
        if self.limit is not None: params['size'] = self.limit
        #   Create the query
        if (query is None) or (filters is None):
            query_filters = self.get_filters()
            _query = query_filters['query']
            _query.update(query or {})
            query = _query
            # --------
            _filters = query_filters['filter']
            _filters.update(filters or {})
            filters = _filters
        #   --------------------------------------------------------------
        LOGGER.info('Looking for {} posts matching {} and {}'.format(
            colored(self.limit or 'all', 'red'),
            colored(pformat(query), 'magenta'),
            colored(pformat(filters), 'magenta'),
        ))
        #   --------------------------------------------------------------
        df = self.session.find_posts(query=query, filters=filters, **params)
        return df

    def get_filters(self):
        res = {
            'query': {},
            'filter': {}
        }
        filters = copy.deepcopy(self.filters)
        randomize = filters.pop('randomize', False)
        _filters = []
        _queries = []
        for k, v in filters.iteritems():
            if k == 'is_labeled':
                fltr = self.get_is_labeled_filter(v)
            elif k in ['date', self.fld_date]:
                k = self.fld_date
                if v:
                    fltr = self.get_date_filter(k, v)
            elif k == 'limit':
                fltr = self.get_limit_filter(v)
            elif k == 'content':
                fltr = self.get_content_query(v)
            elif k == 'status':
                fltr = self.get_status_filter(v)
            elif k == 'community':
                fltr = self.get_community_filter(v)
            elif k == 'word_count':
                fltr = self.get_wordcount_filter(v)
            elif k == 'type':
                fltr = self.get_type_filter(v)
            else:
                fltr = None, None
            _type, value = fltr
            if value:
                res_curr = res[_type]
                if _type == 'filter':
                    _filters.extend([{_k: _v} for _k, _v in value.iteritems()])
                elif _type == 'query':
                    _queries.extend([{_k: _v} for _k, _v in value.iteritems()])
        if _filters:
            if len(_filters) > 1:
                res['filter']['and'] = _filters
            else:
                _k, _v = _filters[0]
                res['filter'][_k] = _v
        if _queries:
            for value in _queries:
                for _k, _v in value.iteritems():
                    res['query'][_k] = _v
        else:
            res['query'] = {'match_all': {}}
        if randomize:
            res['query'] = self.get_random_query(res['query'])
        return res

    def get_type_filter(self, value):
        if value is not None:
            res = {
                "term": {
                    "type": "text"
                }
            }
        else:
            res = {}
        return 'filter', res

    def get_community_filter(self, values):
        if values:
            if not hasattr(values, '__iter__'):
                values = [values]
            res = {
                "terms": {
                    "conversation.community.raw": values
                }
            }
        else:
            res = {}
        return 'filter', res

    def get_status_filter(self, value):
        if value is not None:
            res = {
                "term": {
                    "status": value
                }
            }
        else:
            res = {}
        return 'filter', res

    def get_limit_filter(self, limit):
        res = {}
        if limit is not None:
            try:
                limit = int(limit)
            except (AttributeError, ValueError):
                pass
            else:
                res = {
                    "limit": {
                        "value": limit
                    }
                }
        return 'filter', res

    def get_is_labeled_filter(self, is_labeled):
        #   Get items that have / haven't been labeled yet.
        if is_labeled is True:
            res = {
                "range": {
                    "label_count": {
                        "gte": 1
                    }
                }
            }
        elif is_labeled is False:
            res = {
                "range": {
                    "label_count": {
                        "lt": 1
                    }
                }
            }
        else:
            res = {}
        return 'filter', res

    def get_wordcount_filter(self, value):
        #   Get items that meet or exceed a given word count.
        try:
            value = int(value)
        except (TypeError, ValueError):
            res = {}
        else:
            if value >= 0:
                res = {
                    "range": {
                        "ct_words": {
                            "gte": value
                        }
                    }
                }
        return 'filter', res

    def get_date_filter(self, key, val):
        if val:
            if isinstance(val, basestring):
                val = date_parse(val)
            elif isinstance(val, int) or isinstance(val, float):
                val = datetime.datetime.utcfromtimestamp(val)
            val = val.strftime('%Y%m%dT%H%M%SZ')
            res = {
                "range": {
                    key: {
                        "gte": val
                    }
                }
            }
        else:
            res = {}
        return 'filter', res

    def get_content_query(self, content):
        if content:
            res = {
                "match" : {
                    "content" : content
                }
            }
        else:
            res = {}
        return 'query', res

    def get_random_query(self, query):
        #   Note: Call this last
        if query is None:
            query = {"match_all": {}}
        res = {
            "function_score": {
                "query" : query,
                "random_score" : {}
            }
        }
        return res

    def get_querystring_query(self, query):
        """
        Convert an iterable of queries to a query string:
            dict
                {
                    field: query,
                    ...,
                    field: query
                }
                    ==>
                "field:query ... field:query"

            list
                [ query, ..., query]
                    ==>
                "query ... query"

        """
        if query:
            if isinstance(query, dict):
                query_string = u" ".join(
                    u"{k}:{v}".format(k=k, v=v) for k, v in query.iteritems()
                )
            elif hasattr(query_string, '__iter__'):
                query_string = u" ".join(query)
            else:
                query_string = query
            res = {
                'query_string': {
                    'query': query_string
                }
            }
        else:
            res = {}
        return 'query', res


class ElasticSearchSession(object):
    #: Default host that the index listens on
    host = '127.0.0.1'
    #: Default port that the host listens on
    port = 9200

    def __init__(
            self,
            db_name=None,
            limit=100,
            refresh=False,
            base_query=None,
            doc_type=None,
            **kwargs
        ):
        from elasticsearch import Elasticsearch
        if base_query is None: self.base_query = {'content': '*'}
        #   Initialize the ElasticSearch connections
        self.db_name = db_name or ''
        self.host = kwargs.get('host') or self.port
        self.port = kwargs.get('port') or self.port
        if self.host:
            connection = Elasticsearch([{'host': self.host}])
        else:
            connection = Elasticsearch()
        self.connection = connection

    def find_posts(self, query=None, filters=None, **params):
        limit = params.pop('limit', None)
        if limit: params['size'] = limit
        if filters:
            body = {
                "query": {
                    "filtered": {
                        "filter": filters,
                    }
                }
            }
            if query:
                body['query']['filtered']['query'] = query
        elif query:
            body = {
                "query": query
            }
        else:
            raise Exception("Missing query!")

        posts = self.connection.search(
            index=self.db_name, body=body, **params
        )
        posts = posts['hits']['hits']
        posts = [ p['_source'] for p in posts ]
        idx = None
        for idx, post in enumerate(posts):
            if post.get('labels') is None:
                post['labels'] = []
        #   --------------------------------------------------------------
        LOGGER.info('Found {} posts matching {}'.format(
            colored(idx + 1 if idx is not None else 0, 'red'),
            colored(pformat(body), 'magenta')
        ))
        #   --------------------------------------------------------------
        df = DataFrame(posts)
        return df

    def get_distinct(self, fields, size=100, fltr=None):
        if fltr is None:
            fltr = {
                "match_all": {}
            }
        def get_agg(fld):
            return { "terms": {"field": fld, "size": size}}

        aggs = {
            "aggs": {
                "distinct_values": {
                    "filter" : fltr,
                    "aggs": dict((field, get_agg(field)) for field in fields),
                },
            }
        }
        docs = self.connection.search(
            index=self.db_name, body=aggs, search_type="count"
        )
        docs_agg = docs['aggregations']['distinct_values']
        count = docs_agg.pop('doc_count')
        res = dict(
            (k, dict((b['key'], b['doc_count']) for b in buckets['buckets']))
                for k, buckets in docs_agg.iteritems()
        )
        res['count'] = count
        return res

    def aggregate(
            self, search_type="count", size=None,
        ):
        """
        Run a search query in Elasticsearch (primarily used for aggregations in this module, hence the name, but can be used for generic Elasticsearch search operations).

        :param dict data: The body of the search request
        :param str index: The Elasticsearch index containing the document
        :param str doc_type: The type of the document in the Elasticsearch index
        :param str search_type: The Elasticsearch search_type to use ('count' or 'search', defaults to 'count')
        :param int size: The number of results to return
        :param str host: The host for the Elasticsearch index (defaults to :py:data:`HOST`)
        :param str port: The port for the Elasticsearch index (defaults to :py:data:`PORT`)

        :return: The response from Elasticsearch
        :rtype: dict
        :raises requests.exceptions.HTTPError: If Elasticsearch returns an error response.

        """
        index = self.db_name
        doc_type = self.doc_type
        host = self.host
        port = self.port
        url = 'http://{}:{}/{}/{}/_search'.format(host, port, index, doc_type)
        params = {}
        if search_type is not None:
            params['search_type'] = search_type
        if size is not None:
            params['size'] = size
        data = json.dumps(data)
        resp = requests.post(url, params=params, data=data)
        try:
            resp.raise_for_status()
        except Exception as e:
            msg = textwrap.dedent(
                """
                Error: {}
                Response Status: {}
                Response Reason: {}
                Response Text: {}
                Traceback: {}
                """
            ).format(e, traceback.format_exc(), resp.status_code, resp.reason, resp.text)
            LOGGER.error(msg)
            raise
        docs = resp.json()
        return docs

    def fmt_fields(self, flds):
        fmt = {
            'status': {
                "agg": {
                    "status": {
                        "filter": {
                            "term": {
                                "type": "text"
                            }
                        },
                        "aggs": {
                            "status": {
                                "terms": {
                                    'field': 'status',
                                    'size': 10,
                                }
                            }
                        }

                    }
                },
                "getter": lambda x: x['status']['status']['buckets']
            },
            'community': {
                "agg": {
                    "community": {
                        "filter": {
                            "term": {
                                "type": "text"
                            }
                        },
                        "aggs": {
                            "community": {
                                "terms": {
                                    'field': 'conversation.community.raw',
                                    'size': 100,
                                }
                            }
                        }

                    }
                },
                "getter": lambda x: x['community']['community']['buckets']
            },
            'date': {
                "agg": {
                    "date": {
                        "filter": {
                            "term": {
                                "type": "text"
                            }
                        },
                        "aggs": {
                            "min": {
                                "min": {
                                    "field": "date_create"
                                }
                            },
                            "max": {
                                "max": {
                                    "field": "date_create"
                                }
                            },

                        }
                    }
                },
                "getter": lambda x: {
                    "start": datetime.datetime.utcfromtimestamp(
                        x['date']['min']['value'] / 1000.0
                    ).isoformat(),
                    "end": datetime.datetime.utcfromtimestamp(
                        x['date']['max']['value'] / 1000.0
                    ).isoformat()
                }
            },
            'word_count': {
                "agg": {
                    "word_count": {
                        "filter": {
                            # "match_all": {}
                            "term": {
                                "type": "text"
                            }
                        },
                        "aggs": {
                            "word_count": {
                                "histogram": {
                                    "field": "ct_words",
                                    "interval": 5,
                                }
                            }
                        }
                    }
                },
                "getter": lambda x: x['word_count']['word_count']['buckets'],
            },
        }
        for fld in flds:
            if fmt.get(fld):
                yield fld, fmt.get(fld)

    def get_term_counts_aggs(self, flds):
        res = {}
        flds_fmt = dict(self.fmt_fields(flds))
        for fld, val in flds_fmt.iteritems():
            res.update(val["agg"])
        return flds_fmt, res

    def get_filter_counts(self, flds, fltr=None):
        if fltr is None:
            fltr = {
                "term": {
                    "type": "text"
                }
            }
        if 'date' not in flds:
            flds.append('date')
        flds_fmt, subaggs = self.get_term_counts_aggs(flds)
        aggs = {
            "aggs": {
                "field_counts": {
                    "filter": fltr,
                    "aggs": subaggs
                }
            }
        }
        docs = self.connection.search(
            index=self.db_name, body=aggs, search_type="count"
        )
        docs_agg = docs['aggregations']['field_counts']
        docs_agg['count'] = docs_agg.pop('doc_count')
        for k in docs_agg:
            info = flds_fmt.get(k)
            if info:
                getter = flds_fmt[k]['getter']
                docs_agg[k] = getter(docs_agg)
        return docs_agg

