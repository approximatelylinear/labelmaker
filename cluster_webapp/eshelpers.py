
import os
import datetime
import traceback
import pdb
from pprint import pformat

import requests

def try_request(url, method, logger=None, **params):
    if not callable(method):
        method = getattr(requests, method)
    try:
        resp = method(url, **params)
    except Exception as e:
        log_error_request(url, e, logger)
    else:
        try:
            resp.raise_for_status()
        except Exception as e:
            log_error_request(url, e, logger)
        else:
            return resp


def log_error_request(url, e, logger=None):
    text_resp = (
        e.response.text if (
            hasattr(e, 'response') and hasattr(e.response, 'text')
        ) else '(unavailable)'
    )
    msg = u"""
    Error getting [ {u} ]
    Error: {e}
    Response: {r}
    Traceback: {t}
    """.format(u=url, e=e, r=text_resp, t=traceback.format_exc())
    msg = textwrap.dedent(msg)
    if logger is None:
        print msg
    else:
        logger.error(msg)



def fmt_conversation(doc_content):
    if 'conversation' in doc_content:
        if 'date_create' in doc_content['conversation'] and (doc_content['conversation']['date_create'] is None or (doc_content['conversation']['date_create'] == 'None')):
            doc_content['conversation']['date_create'] = '19700101T000000Z'
            doc_content['conversation']['timestamp'] = '19700101T000000Z'
        for k, v in doc_content['conversation'].iteritems():
            if k.startswith('ct_') and (v is None or v == 'None'):
                doc_content['conversation'][k] = 0
    if 'meta' in (doc_content.get('conversation') or {}):
        if isinstance(doc_content['conversation']['meta'], dict):
            del doc_content['conversation']['meta']
    return doc_content


def fmt_bulk_request(docs, formatters=None):
    """
    Pattern:
        { "index" : { "_index" : "test", "_type" : "type1", "_id" : "1" } }
        { "field1" : "value1" }
        { "delete" : { "_index" : "test", "_type" : "type1", "_id" : "2" } }
        { "create" : { "_index" : "test", "_type" : "type1", "_id" : "3" } }
        { "field1" : "value3" }
        { "update" : {"_id" : "1", "_type" : "type1", "_index" : "index1"} }
        { "doc" : {"field2" : "value2"} }

    Note: Sequence must end in a newline.
    """
    if formatters is None: formatters = []
    dts = Counter()
    def to_json(docs):
        for info in docs:
            ##################
            LOGGER_MOD.debug(pformat(info))
            # pdb.set_trace()
            ##################
            action = info['action']
            doc = info['data']
            doc_content = doc
            if action == 'update':
                if 'doc' in doc:
                    #   Remove _id from update content.
                    doc_content = doc['doc']
            ##################
            if '_id' not in doc_content:
                continue
            ##################
            _id = doc_content['_id']
            if action in ['update', 'delete']:
                del doc_content['_id']
            ####################
            # print pformat(doc)
            # pdb.set_trace()
            ####################
            for formatter in formatters:
                doc_content = formatter(doc_content)
            #   e.g.
            #       The type and index will be specified in the url.
            #       { "update" : {"_id" : "1"}
            #       { "doc" : {"field2" : "value2"} }
            yield json.dumps({action: dict(_id=_id)})
            #   Handle for non-json objs like datetimes.
            yield json.dumps(serialize_for_json(doc))
    docs_fmt = to_json(docs)
    docs_json = u'\n'.join(docs_fmt) + u'\n'
    return docs_json



def to_elasticsearch(
        docs, index, doc_type, hosts=HOSTS, get_count=False, **kwargs
    ):
    formatters = kwargs.pop('formatters', None)
    ###########
    if hasattr(docs, 'next'):
        docs = list(docs)
    LOGGER_MOD.debug(pformat(docs))
    # pdb.set_trace()
    ###########
    docs_by_id = dict(
        (
            (
                doc['data']['doc']['_id']
                    if 'doc' in doc['data'] else doc['data']['_id']
            ),
            doc['data']
        )
            for doc in copy.deepcopy(docs)
    )
    if docs:
        docs_json = []
        docs = [
            {
                'action': info['action'],
                'data': (
                    sum_previous(info['data'], index, doc_type, hosts)
                        if info['action'] == 'update'
                    else info['data']
                )
            }
                for info in docs
        ]
        docs_json = fmt_bulk_request(docs, formatters=formatters)
        if docs_json:
            ###########
            LOGGER_MOD.debug("Docs:\n{}".format(docs_json))
            # pdb.set_trace()
            ###########
            for host in hosts:
                inserts = _to_elasticsearch_bulk(
                    docs_json, host, index, doc_type, get_count
                )
                retry_as_insert(
                    docs_by_id, inserts, host, index, doc_type, get_count
                )


def _to_elasticsearch_single(doc, host, index, doc_type, get_count, action):
    #######
    # pdb.set_trace()
    #######
    _id = urllib.quote(doc['_id'])
    url = 'http://{}:9200/{}/{}/{}'.format(host, index.lower(), doc_type, _id)
    if action == 'update':
        url += '/_update'
    inserts = _to_elasticsearch(
        json.dumps(doc), url, host, index, doc_type, get_count
    )
    return inserts

def _to_elasticsearch_bulk(docs_json, host, index, doc_type, get_count):
    url = 'http://{}:9200/{}/{}/_bulk'.format(host, index.lower(), doc_type)
    inserts = _to_elasticsearch(docs_json, url, host, index, doc_type, get_count)
    return inserts



def get_upserts(doc_resp):
    updates = (
        item.get('update')
            for item in (doc_resp.get('items') or [])
    )
    updates = (
        item for item in updates if item and item.get('error')
    )
    updates = (
        item for item in updates if item.get('status') == 404
    )
    upserts = [ item['_id'] for item in updates ]
    return upserts


def _to_elasticsearch(
        data, url, host, index, doc_type, refresh=True, get_count=False
    ):
    tries = 10
    wait = 1
    upserts = None
    ##############
    # pdb.set_trace()
    ##############
    #   =====================================================
    def handle_exception(url, e, tries, wait):
        log_error_request(url, e)
        tries -= 1
        wait *= 2
        LOGGER_MOD.info("Waiting {} seconds...".format(wait))
        time.sleep(wait)
        return tries, wait
    #   =====================================================
    while tries:
        try:
            resp = requests.post(url, data=data)
        except Exception as e:
            tries, wait = handle_exception(url, e, tries, wait)
            continue
        else:
            try:
                resp.raise_for_status()
            except Exception as e:
                tries, wait = handle_exception(url, e, tries, wait)
                continue
            else:
                tries = 0
                doc_resp = resp.json()
                if doc_resp.get('errors'):
                    LOGGER_MOD.error(pformat(doc_resp))
                    ###############
                    # pdb.set_trace()
                    ###############
                    upserts = get_upserts(doc_resp)
            if refresh:
                url = 'http://{}:9200/{}/_refresh'.format(host, index.lower())
                try:
                    resp = requests.post(url)
                except Exception as e:
                    log_error_request(url, e)
            if get_count:
                url_ct = 'http://{}:9200/{}/{}/_count'.format(
                    host, index.lower(), doc_type
                )
                try:
                    resp_count = requests.get(url_ct, params=dict(q='*'))
                except Exception as e:
                    log_error_request(url, e)
                else:
                    doc_resp_count = resp_count.json()
                    LOGGER_MOD.info(
                        "Total docs: {0}".format(doc_resp_count['count'])
                    )
            break
    return upserts


def to_elasticsearch_temp(docs, index, doc_type):
    ##########
    #   Only ever upload comments
    doc_type = 'comment'
    ##########
    path_base = os.path.join(PATH_ES_TEMP, 'new', index, doc_type)
    if not os.path.exists(path_base): os.makedirs(path_base)
    now = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S%fZ')
    len_docs = 'unknown' if hasattr(docs, 'next') else len(docs)
    fname = '{}_{}.json'.format(now, len_docs)
    path = os.path.join(path_base, fname)
    with open(path, 'wb') as f:
        LOGGER_MOD.info("Saving {} docs to {}".format(len_docs, path))
        for doc in docs:
            f.write(json.dumps(doc) + '\n')


def to_elasticsearch_from_file(hosts=HOSTS, get_count=False, **kwargs):
    path_base_new = os.path.join(PATH_ES_TEMP, 'new')
    path_base_old = os.path.join(PATH_ES_TEMP, 'old')
    indexes = [ b for b in os.listdir(path_base_new) if not b.startswith('.') ]
    for index in indexes:
        doc_types = os.listdir(os.path.join(path_base_new, index))
        doc_types = [ dt for dt in doc_types if not dt.startswith('.') ]
        for doc_type in doc_types:
            fnames = os.listdir(os.path.join(path_base_new, index, doc_type))
            fnames = [ fname for fname in fnames if not fname.startswith('.') ]
            path_new = os.path.join(path_base_new, index, doc_type)
            path_old = os.path.join(path_base_old, index, doc_type)
            if not os.path.exists(path_old): os.makedirs(path_old)
            for fname in fnames:
                num_docs = fname.split('_')[-1].strip('.json')
                # num_docs = int(num_docs)
                path = os.path.join(path_new, fname)
                #########
                print "Loading {}".format(path)
                #########
                with open(path, 'rbU') as f:
                    docs = [ json.loads(l.strip()) for l in f if l.strip() ]
                    try:
                        to_elasticsearch(
                            docs, index, doc_type,
                            hosts=hosts, get_count=get_count, **kwargs
                        )
                    except Exception as e:
                        pass
                    else:
                        shutil.move(
                            os.path.join(path_new, fname),
                            os.path.join(path_old, fname)
                        )


def sum_previous(doc, index, doc_type, hosts=HOSTS):
    _id = doc.get('_id') or doc['doc'].get('_id')
    if _id:
        add = doc['doc'].pop('add', None)
        if add:
            #   Sum current with previous. (Use first host for now.)
            url = 'http://{}:9200/{}/{}/{}'.format(
                hosts[0], index.lower(), doc_type, urllib.quote(_id)
            )
            resp = requests.get(url)
            try:
                resp.raise_for_status()
            except Exception as e:
                log_error_request(e, url)
            else:
                doc_resp = resp.json()
                data_prev = doc_resp.get('_source') or {}
                for k in add:
                    doc['doc'][k] += data_prev.get(k) or 0
    return doc



def retry_as_insert(docs_by_id, docs_json, host, index, doc_type, get_count):
    #   Index failed updates.
    docs_json_ins = []
    for _id in docs_json:
        doc = docs_by_id[_id]['doc']
        ###############
        LOGGER_MOD.debug(u"{}, {}".format(_id, pformat(doc)))
        # pdb.set_trace()
        ###############
        doc['_id'] = _id
        docs_json_ins.append(
            '\n'.join([
                json.dumps({'index': dict(_id=_id)}),
                json.dumps(doc),
            ])
        )
    if docs_json_ins:
        docs_json_ins = '\n'.join(docs_json_ins) + '\n'
        _to_elasticsearch_bulk(
            docs_json_ins, host, index, doc_type, get_count
        )
