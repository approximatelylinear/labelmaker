
import os
import pdb
import datetime
from pprint import pformat
import logging

import tornado.ioloop
import tornado.web
import tornado.httpserver
from tornado.escape import json_decode, json_encode
from tornado.options import define, options, parse_command_line
from werkzeug import secure_filename
from termcolor import colored

from models import models, utils as model_utils
from models import models_metadata
import conf

LOGGER = logging.getLogger(__name__)


class ExportHandler(tornado.web.RequestHandler):
    def get(self, fname):
        #   '/cluster/download/<fname>'
        #   Try to download the file.
        fname = secure_filename(fname)
        path = os.path.join(conf.DOWNLOAD_PATH, fname)
        if not os.path.isfile(path):
            raise tornado.web.HTTPError(404)
        info = os.stat(path)
        mimetype = 'text/csv'   #   'application/octet-stream'   #      "application/unknown"
        self.set_header("Content-Type", 'application/octet-stream')
        self.set_header("Content-Length", info.st_size)
        self.set_header(
            "Last-Modified",
            datetime.datetime.utcfromtimestamp(info.st_mtime)
        )
        self.set_header('Content-Disposition', 'attachment; filename={0}'.format(path))
        with open(path, 'rbU') as f:
            self.finish(f.read())

    def post(self, cluster_id=None):
        req_data = json_decode(self.request.body)
        try:
            df, fname = models.ClusterTable.export(
                h5fname = req_data['h5fname'],      #   OS location of h5 db
                group_path=req_data['group_path'],  #   h5 path
                # TBD: This does nothing right now...
                db_name=req_data['db_name'],  #   Database name
            )
        except Exception as e:
            LOGGER.exception(e)
            result = {'error': 1, 'msg': repr(e)}
        else:
            result = {'error': 0, 'fname': fname}
        result = json_encode(result)
        self.set_header("Content-Type", "application/json")
        self.write(result)



class RemoveRowHandler(tornado.web.RequestHandler):
    def post(self, cluster_id=None):
        #   cluster_id
        req_data = tornado.escape.json_decode(self.request.body)
        idx = int(req_data['idx'])
        try:
            models.ClusterTable.load(
                h5fname = req_data['h5fname'],
                group_path=req_data['group_path'],
                table_name=req_data['table_name'],
                callback=models.ClusterTable.remove_table_row,
                idx=idx
            )
        except Exception as e:
            LOGGER.exception(e)
            result = {
                'error' : 1,
                'msg'   : repr(e)
            }
        else:
            result = {}
        result = json_encode(result)
        self.set_header("Content-Type", "application/json")
        self.write(result)



class ClusterHandler(tornado.web.RequestHandler):
    def post(self, cluster_id=None):
        req_data = tornado.escape.json_decode(self.request.body)
        #   -------------------
        LOGGER.debug(pformat(req_data))
        #   -------------------
        table_name = req_data['table_name']         #   table name
        group_path = req_data['group_path']         #   h5 path
        h5fname = req_data['h5fname']                 #   OS location of h5 db
        num_clusters = int(req_data['numClusters'])
        refresh = req_data['refresh']               #   Whether to redo the clustering
        cluster_method = req_data.get('cluster_method', None) #   Name of clustering method
        db_name = req_data['db_name'] or None
        #   Retrieve cluster from h5 store.
        cluster_maker = models.Cluster(
            table_name=table_name,
            group_path=group_path,
            num_clusters=num_clusters,
            refresh=refresh,
            cluster_method=cluster_method,
            h5fname=h5fname,
            db_name=db_name,
        )
        cluster_maker.load()
        clusters = cluster_maker.cluster()
        children = cluster_maker.to_h5(clusters)
        #   return sample of data to app and info on children
        #   children: <list> of <child>
        #       child:
        #           sample      <df>
        #           path        <str>
        #           name        <str>
        #           h5fname     <str>
        #           id          <str>
        #           size        <int>
        #           tags        <list>
        #           info_feats  <list of dict>
        clusters = model_utils.sample(clusters, 1000)
        clusters = clusters.drop(['clean_tokens'], axis=1)
        parent_data = [ row.to_dict() for _, row in clusters.iterrows() ]
        for child in children:
            sample = child.pop('sample').drop(['clean_tokens'], axis=1)
            #   Convert dataframes to lists of dictionaries.
            child['data'] = [ row.to_dict() for _, row in sample.iterrows() ]
        result = dict(parent=parent_data, children=children)
        result = json_encode(result)
        self.set_header("Content-Type", "application/json")
        self.write(result)




class ClusterDataHandler(tornado.web.RequestHandler):
    def delete(self, cluster_id):
        # delete a single cluster
        try:
            models.ClusterTable.load_and_remove_from_id(cluster_id)
        except Exception as e:
            #################
            LOGGER.exception(e)
            #################
            result = {'error': 1}
        else:
            result = {}
        result = json_encode(result)
        self.set_header("Content-Type", "application/json")
        self.write(result)

    def put(self, cluster_id=None):
        # update a single cluster
        cluster_id = self.get_argument('cluster_id', '')
        req_data = tornado.escape.json_decode(self.request.body)
        updated = {}
        if 'tags' in req_data:
            result = models.ClusterTable.update_labels(
                labels=req_data['tags'],
                h5fname = req_data['h5fname'],
                group_path=req_data['group_path'],
                table_name=req_data['table_name'],
                #   Modify the models.models_h5.ClusterTable.add_labels method
                # db_name=req_data.get('db_name'), # TBD
                # db_type=req_data.get('db_type'), # TBD
            )
            updated['tags'] = result or []
        #   ------------------------------------------------------------
        LOGGER.info('Updated tags: {}'.format(colored(pformat(updated), 'magenta')))
        #   ------------------------------------------------------------
        result = json_encode(updated)
        self.set_header("Content-Type", "application/json")
        self.write(result)



class RawDataHandler(tornado.web.RequestHandler):
    base_path = conf.RAW_DATA_PATH

    def list_files(self):
        #   List files in 'raw_data'
        raw_data_files = [(n, None) for n in  os.listdir(self.base_path)]
        raw_data_files.extend(models_metadata.list_datasets())
        return raw_data_files

    def get(self):
        #   List raw data files
        raw_data_files = self.list_files()
        raw_data_files = [ n for n, pk in raw_data_files ]
        result = {'fnames': raw_data_files}
        result = json_encode(result)
        self.set_header("Content-Type", "application/json")
        self.write(result)

    def post(self):
        #   Load raw data file
        req_data = tornado.escape.json_decode(self.request.body)
        source_name = req_data['sourcename']
        refresh = req_data['refresh']
        columns = req_data.get('columns', None)
        #   Get db info from sql store.
        session = models_metadata.SESSION()
        raw_data_files = self.list_files()
        names_2_pk = dict(raw_data_files)
        assert source_name in names_2_pk
        pk = names_2_pk[source_name]
        obj = session.query(models_metadata.Dataset).get(pk) if pk else None
        if obj:
            db_name, df_cls = obj.name_db, obj.manager
            df_cls = getattr(models, df_cls.split('.')[-1])
            has_user_data = req_data.get('has_user_data', False)
            if has_user_data:
                #   Get filters to restrict data by.
                filters = req_data.get('filters', {})
                filters = [
                    (k[:-len('_filter')], v) for k, v in filters.iteritems()
                ]
                filters = dict(filters)
                is_unlabeled = filters.pop('isUnlabeled', None)
                is_labeled = (not is_unlabeled) if is_unlabeled else None
                filters['is_labeled'] = is_labeled
                #   Get fields
                limit = filters.pop('sampleSize') or 1000
                limit = int(limit)
                filters['limit'] = limit
                #   Get fields to include in output
                fields = req_data.get('fields', {})
                #   Remove 'field' suffix
                fields = [ k[:-6] for k, v in fields.iteritems() if v ]
                df_maker = df_cls(
                    db_name,
                    refresh=refresh,
                    filters=filters,
                    fields=fields
                )
            else:
                #   Get more data from the user
                df_maker = df_cls(db_name, refresh=refresh,)
                db_info = df_maker.get_info()
                """
                Expected format of db_info:
                    [<choice>, ..., <choice]
                    OR
                    [{'key': <key>, 'value': <value>}, ..., {'key': <key>, 'value': <value>}]
                """
                result = json_encode({'db_info': db_info})
                #########
                LOGGER.debug(pformat(db_info))
                #########
                self.set_header("Content-Type", "application/json")
                self.write(result)
                return
                ########
                #   EXIT
                ########
        else:
            df_maker = models.ClusterDataFrame(
                source_name, refresh=refresh, columns=columns
            )
        df = df_maker.load()
        #   return sample of data to app and info on children
        #   children: <list> of <child>
        #       child:
        #           sample      <df>
        #           path        <str>
        #           name        <str>
        #           h5fname     <str>
        #           id          <str>
        #           size        <int>
        #           tags        <list>
        children = df_maker.to_h5()
        for child in children:
            sample = child.pop('sample').drop(['clean_tokens'], axis=1)
            #   Convert dataframes to lists of dictionaries.
            child['data'] = [ row.to_dict() for _, row in sample.iterrows() ]
        result = json_encode({'children': children})
        self.set_header("Content-Type", "application/json")
        self.write(result)


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('index.html')



define("host", default='127.0.0.1', help="run on the given host", type=str)
define(
    "port",
    default=5001, #8888,
    help="run on the given port", type=int
)

ROUTES = [
    (r"^/?$", MainHandler),
    (r"^/load/?$", RawDataHandler),
    (r"^/cluster/?$", ClusterDataHandler),
    (r"^/cluster/(?P<cluster_id>[\w\-\.]+)/?$", ClusterDataHandler),
    (r"^/cluster/(?P<cluster_id>[\w\-\.]+)/cluster/?$", ClusterHandler),
    (r"^/cluster/(?P<cluster_id>[\w\-\.]+)/export/?$", ExportHandler),
    (r"^/cluster/(?P<fname>[\w\-\.]+)/download/?$", ExportHandler),
    (r"^/cluster/(?P<cluster_id>[\w\-\.]+)/remove_row/?$", RemoveRowHandler),
    #   TODO:
    #       Add classification handler

]

def tornado_main():
    parse_command_line()
    app = tornado.web.Application(
        ROUTES,
        default_host=options.host,
        template_path=os.path.join(os.path.dirname(__file__), "frontend", "templates"),
        static_path=os.path.join(
            os.path.dirname(__file__), "frontend", "static"),
    )
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    LOGGER.info("Starting to listen on {}:{}".format(options.host, options.port))
    tornado.ioloop.IOLoop.instance().start()


if __name__ == "__main__":
    import logging
    # config = conf.setup_logging()
    logging.basicConfig(level=logging.DEBUG)
    tornado_main()
