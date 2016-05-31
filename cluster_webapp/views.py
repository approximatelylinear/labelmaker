
import os
import pdb
import datetime
from pprint import pformat

import tornado.ioloop
import tornado.web
import tornado.httpserver
from tornado.escape import json_decode, json_encode
from tornado.options import define, options, parse_command_line
from werkzeug import secure_filename
from termcolor import colored

import models
import models_metadata
import conf

#:  The directory containing this script
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

DOWNLOAD_FOLDER = os.path.join(THIS_DIR, 'data', 'clusters', 'labeled')
ALLOWED_EXTENSIONS = set(['txt', 'csv', 'xlsx', 'xls'])


class ExportHandler(tornado.web.RequestHandler):
    def get(self, fname):
        #   '/cluster/<fname>/download'
        #   Try to download the file.
        ###########
        # print 'download_file: {0}'.format(fname)
        # pdb.set_trace()
        ###########
        fname = secure_filename(fname)
        path = os.path.join(DOWNLOAD_FOLDER, fname)
        if not os.path.isfile(path):
            raise tornado.web.HTTPError(404)
        info = os.stat(path)
        mimetype = 'text/csv'   #   'application/octet-stream'   #      "application/unknown"
        self.set_header("Content-Type", 'application/octet-stream')
        self.set_header("Content-Length", info.st_size)
        """
        self.set_header("Last-Modified", datetime.datetime.utcfromtimestamp(info.st_mtime))
        """
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
                mongo_db_name=req_data['db_name'],  #   Mongo database name
            )
        except Exception as e:
            print e
            result = {'error': 1, 'msg': repr(e)}
        else:
            result = {'error': 0, 'fname': fname}
        result = json_encode(result)
        self.set_header("Content-Type", "application/json")
        self.write(result)



class RowHandler(tornado.web.RequestHandler):
    def post(self, cluster_id=None):
        #   cluster_id
        req_data = tornado.escape.json_decode(self.request.body)
        #   -------------------
        # print pformat(req_data)
        # pdb.set_trace()
        #   -------------------
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
            print e
            # pdb.set_trace()
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
        # print pformat(req_data)
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
        clusters = models.ClusterDataFrame.sample(clusters, 1000)
        clusters = clusters.drop(['clean_tokens'], axis=1)
        parent_data = [ row.to_dict() for _, row in clusters.iterrows() ]
        for child in children:
            sample = child.pop('sample').drop(['clean_tokens'], axis=1)
            #   Convert dataframes to lists of dictionaries.
            child['data'] = [ row.to_dict() for _, row in sample.iterrows() ]
        result = dict(parent=parent_data, children=children)
        result = json_encode(result)
        #################
        # pdb.set_trace()
        ################
        self.set_header("Content-Type", "application/json")
        self.write(result)




class ClusterDataHandler(tornado.web.RequestHandler):
    def delete(self, cluster_id):
        # delete a single cluster
        try:
            models.ClusterTable.load_and_remove_from_id(cluster_id)
        except Exception as e:
            #################
            print e
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
                db_name=req_data.get('db_name'), # TBD
                db_type=req_data.get('db_type'), # TBD
            )
            updated['tags'] = result
        #   ------------------------------------------------------------
        print 'Updated tags: {0}'.format(colored(pformat(updated), 'magenta'))
        #   ------------------------------------------------------------
        result = json_encode(updated)
        self.set_header("Content-Type", "application/json")
        self.write(result)



class RawDataHandler(tornado.web.RequestHandler):
    base_path = os.path.join(THIS_DIR, 'data', 'raw')
    # db_names = {
    #     'Rock the Ranch'            : ['pm_comments_rtr', models.PMClusterDataFrame],
    #     'All PM Comments'           : ['pm_web_comments', models.PMClusterDataFrame],
    #     'PM Comments (RATED)'       : ['pm_web_comments_rated', models.PMClusterDataFrame],
    #     'Hallmark Boardreader'      : ['boardreader_hallmark', models.BRClusterDataFrame],
    #     'Hallmark Twitter'          : ['topsy_hallmark', models.TopsyClusterDataFrame],
    #     'Norton Twitter'            : ['topsy_norton', models.TopsyClusterDataFrame],
    #     'Samsung Galaxy Twitter'    : ['topsy_samsung_galaxy', models.TopsyClusterDataFrame],
    #     'Alternative Tobacco'       : ['topsy_alternative_tobacco', models.TopsyClusterDataFrame],
    #     'Insurance Agent Twitter'   : ['topsy_insurance_agent', models.TopsyClusterDataFrame],
    #     'Insurance Agent Blog/Board'   : ['boardreader_insurance_agent', models.BRClusterDataFrame],
    #     'McDonalds Books Twitter'   : ['topsy_mcdonalds_books_weekof', models.TopsyClusterDataFrame],
    #     'Meanstinks Twitter'        : ['topsy_meanstinks_big_asmbly', models.TopsyClusterDataFrame],
    #     'McDonalds Flavor Twitter'  : ['topsy_mcdonalds_flavor', models.TopsyClusterDataFrame],
    #     'McDonalds Employee Twitter'  : ['topsy_mcdonalds_employee', models.TopsyClusterDataFrame],
    #     'McDonalds From Employee Twitter'  : ['topsy_mcdonalds_from_employee_2', models.TopsyClusterDataFrame],
    # }
    # datasources_2_managers = {
    #     'Rock the Ranch'            : ['pm_comments_rtr', models.PMClusterDataFrame],
    #     'All PM Comments'           : ['pm_web_comments', models.PMClusterDataFrame],
    #     'PM Comments (RATED)'       : ['pm_web_comments_rated', models.PMClusterDataFrame],
    #     'Hallmark Boardreader'      : ['boardreader_hallmark', models.BRClusterDataFrame],
    #     'Hallmark Twitter'          : ['topsy_hallmark', models.TopsyClusterDataFrame],
    #     'Norton Twitter'            : ['topsy_norton', models.TopsyClusterDataFrame],
    #     'Samsung Galaxy Twitter'    : ['topsy_samsung_galaxy', models.TopsyClusterDataFrame],
    #     'Alternative Tobacco'       : ['topsy_alternative_tobacco', models.TopsyClusterDataFrame],
    #     'Insurance Agent Twitter'   : ['topsy_insurance_agent', models.TopsyClusterDataFrame],
    #     'Insurance Agent Blog/Board'   : ['boardreader_insurance_agent', models.BRClusterDataFrame],
    # }
    # datasources = [
    #     {
    #         'name'      : 'Allstate',
    #         'db_name'   : '',
    #         'children'  : [
    #             {
    #                 'name'      : 'Insurance Agent',
    #                 'db_name'   : '',
    #                 'fts_url'   : '',
    #                 'children'  : [
    #                     {
    #                         'id'        : 'allstate_insurance_agent_tweets',
    #                         'name'      : 'Tweets',
    #                         'db_name'   : 'topsy_insurance_agent',
    #                         'fts_url'   : '',
    #                         'children'  : []
    #                     },
    #                     {
    #                         'id'        : 'allstate_insurance_agent_blog/board_posts',
    #                         'name'      : 'Blog/Board Posts',
    #                         'db_name'   : 'boardreader_insurance_agent',
    #                         'fts_url'   : '',
    #                         'children'  : []
    #                     }
    #                 ]
    #             },
    #         ]
    #     },
    #     {
    #         'name'      : 'Hallmark',
    #         'db_name'   : '',
    #         'children'  : [
    #             {
    #                 'id'        : 'hallmark_tweets',
    #                 'name'      : 'Tweets',
    #                 'db_name'   : 'topsy_hallmark',
    #                 'fts_url'   : '',
    #                 'children'  : []
    #             },
    #             {
    #                 'id'        : 'hallmark_blog/board_posts',
    #                 'name'      : 'Blog/Board Posts',
    #                 'db_name'   : 'boardreader_hallmark',
    #                 'fts_url'   : '',
    #                 'children'  : []
    #             }
    #         ]
    #     },
    #     {
    #         'name'      : 'Norton',
    #         'db_name'   : '',
    #         'children'  : [
    #             {
    #                 'id'        : 'norton_tweets',
    #                 'name'      : 'Tweets',
    #                 'db_name'   : 'topsy_norton',
    #                 'fts_url'   : '',
    #                 'children'  : []
    #             },
    #         ]
    #     },
    #     {
    #         'name'      : 'PM',
    #         'db_name'   : '',
    #         'children'  : [
    #             {
    #                 'id'        : 'pm_all_comments',
    #                 'name'      : 'All Comments',
    #                 'db_name'   : 'pm_web_comments',
    #                 'fts_url'   : '',
    #                 'children'  : []
    #             },
    #             {
    #                 'id'        : 'pm_rated_comments',
    #                 'name'      : 'Rated Comments',
    #                 'db_name'   : 'pm_web_comments_rated',
    #                 'fts_url'   : '',
    #                 'children'  : []
    #             },
    #             {
    #                 'id'        : 'pm_rock_the_ranch_comments',
    #                 'name'      : 'Rock the Ranch Comments',
    #                 'db_name'   : 'pm_comments_rtr',
    #                 'fts_url'   : '',
    #                 'children'  : []
    #             },
    #             {
    #                 'id'        : 'pm_alternative_tobacco_tweets',
    #                 'name'      : 'Alternative Tobacco Tweets',
    #                 'db_name'   : 'topsy_alternative_tobacco',
    #                 'fts_url'   : '',
    #                 'children'  : []
    #             }
    #         ]
    #     },
    #     {
    #         'name'      : 'Samsung',
    #         'db_name'   : '',
    #         'children'  : [
    #             {
    #                 'id'        : 'samsung_galaxy_tweets',
    #                 'name'      : 'Galaxy Tweets',
    #                 'db_name'   : 'topsy_samsung_galaxy',
    #                 'fts_url'   : '',
    #                 'children'  : []
    #             },
    #         ]
    #     },
    # ]

    def list_files(self):
        #   List files in 'raw_data'
        # raw_data_files = [(n, None) for n in  os.listdir(self.base_path)]
        raw_data_files = []
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
        # pk = names_2_pk[source_name]
        # obj = session.query(models_metadata.Dataset).get(pk) if pk else None
        if obj:
            db_name, df_cls = obj.name_db, obj.manager
            df_cls = getattr(models, df_cls.split('.')[-1])
            #########
            # print db_name, df_cls
            # pdb.set_trace()
            #########
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
                # print pformat(fields)
                # pdb.set_trace()
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
                # print pformat(db_info)
                # pdb.set_trace()
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
    default=5000, #8888,
    help="run on the given port", type=int
)

ROUTES = [
    (r"^/?$", MainHandler),
    (r"^/load/?$", RawDataHandler),
    (r"^/cluster?/$", ClusterDataHandler),
    (r"^/cluster/(?P<cluster_id>[\w\-\.]+)/$", ClusterDataHandler),
    (r"^/cluster/(?P<cluster_id>[\w\-\.]+)/cluster/$", ClusterHandler),
    (r"^/cluster/(?P<cluster_id>[\w\-\.]+)/export/$", ExportHandler),
    (r"^/cluster/(?P<fname>[\w\-\.]+)/download/$", ExportHandler),
    (r"^/cluster/(?P<cluster_id>[\w\-\.]+)/remove_row/$", RowHandler),
    #   TODO:
    #       Add classification handler

    #   TODO:   Migrate to this url pattern:
    #               (r"/cluster/cluster/<cluster_id>", ClusterHandler),
    #               (r"/cluster/export/<cluster_id>", ExportHandler),
    #               (r"/cluster/remove_row/<cluster_id>", RowHandler),
]

def tornado_main():
    parse_command_line()
    app = tornado.web.Application(
        ROUTES,
        default_host=options.host,
        # cookie_secret="__TODO:_GENERATE_YOUR_OWN_RANDOM_VALUE_HERE__",
#         login_url="/auth/login",
        template_path=os.path.join(os.path.dirname(__file__), "templates"),
        static_path=os.path.join(os.path.dirname(__file__), "static"),
        # xsrf_cookies=True,
    )
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    #####
    print "Starting to listen on {}:{}".format(options.host, options.port)
    tornado.ioloop.IOLoop.instance().start()


if __name__ == "__main__":
    import logging
    config = conf.setup_logging()
    logging.config.dictConfig(config)
    tornado_main()
