
from flask import Flask, render_template, request, jsonify, json, send_from_directory
from flask.views import MethodView
from werkzeug import secure_filename
from termcolor import colored

import models

#:  The directory containing this script
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


DEBUG = True
SECRET_KEY = 'abcdefg'
DOWNLOAD_FOLDER = os.path.join(THIS_DIR, 'data', 'clusters', 'labeled')
# DOWNLOAD_FOLDER = '/Users/mberends/Desktop/cluster_webapp/data/clusters/labeled'
ALLOWED_EXTENSIONS = set(['txt', 'csv', 'xlsx', 'xls'])
# SERVER_NAME = '10.3.50.127:5000' # '0.0.0.0:5000'  #   '127.0.0.1:5500'

#   Create the application using the above settings
app = Flask(__name__)
app.config.from_object(__name__)



# class UserAPI(MethodView):

#     def get(self, user_id):
#         if user_id is None:
#             # return a list of users
#             pass
#         else:
#             # expose a single user
#             pass

#     def post(self):
#         # create a new user
#         pass

#     def delete(self, user_id):
#         # delete a single user
#         pass

#     def put(self, user_id):
#         # update a single user
#         pass


# app.add_url_rule('/about', view_func=RenderTemplateView.as_view(
#     'about_page', template_name='about.html'))


# app.add_url_rule(
#     '/api', 
#     view_func=index.as_view(
#         'index', 
#         template_name='index.html')
# )



@app.route('/api/')
def index():
    return render_template('index.html')
    


@app.route('/api/clusters/<cluster_id>/export', methods=['GET', 'POST'])
def export(cluster_id):
    req_data = request.get_json()
    if request.method == 'POST':
        #   -------------------
        # print pformat(req_data)
        # pdb.set_trace()
        #   -------------------
        try:
            df, fname = models.ClusterTable.export(
                h5fname = req_data['h5fname'],      #   OS location of h5 db
                group_path=req_data['group_path'],  #   h5 path
            )
        except Exception as e:
            print e
            # pdb.set_trace()
            result = {'error': 1}
        else:
            result = {'fname': os.path.join('/api/data', fname)}
        return jsonify(result)


@app.route('/api/data/<fname>')
def download_file(fname):
    ###########
    # print 'download_file: {0}'.format(fname)
    ###########
    fname = secure_filename(fname)
    return send_from_directory(
        app.config['DOWNLOAD_FOLDER'],
        fname
    )


@app.route('/api/clusters/<cluster_id>/remove_row', methods=['GET', 'POST'])
def remove_row(cluster_id):
    req_data = request.get_json()
    if request.method == 'POST':
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
            result = {'error': 1}
        else:
            result = {}
        return jsonify(result)



@app.route('/api/clusters/<cluster_id>/cluster', methods=['GET', 'POST'])
def cluster(cluster_id):
    req_data = request.get_json()
    if request.method == 'POST':
        #   -------------------
        # print pformat(req_data)
        #   -------------------
        table_name = req_data['table_name']         #   table name
        group_path = req_data['group_path']         #   h5 path
        h5fname = req_data['h5fname']                 #   OS location of h5 db
        num_clusters = int(req_data['numClusters'])
        refresh = req_data['refresh']               #   Whether to redo the clustering    
        cluster_method = req_data.get('cluster_method', None) #   Name of clustering method
        #   Retrieve cluster from h5 store.
        cluster_maker = models.Cluster(
            table_name=table_name, 
            group_path=group_path, 
            num_clusters=num_clusters, 
            refresh=refresh, 
            cluster_method=cluster_method, 
            h5fname=h5fname
        )
        cluster_maker.load()
        clusters = cluster_maker.cluster()
        children = cluster_maker.to_h5(clusters)
        # if children:
        #     #   Get features characterizing each cluster.
        #     info_feats = cluster_maker.get_info_features()
        #     #   Save the features.
        #     models.ClusterTable.load(
        #         h5fname = children[0].loc[0]['h5fname'],
        #         group_path=req_data['group_path'],
        #         table_name=req_data['table_name'],
        #         callback=models.ClusterTable.remove_table_row,
        #         idx=idx
        #     )

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
        #####
        # print pformat(children)
        # pdb.set_trace()
        ######
        response = jsonify(parent=parent_data, children=children)
        #####
        # print response
        # pdb.set_trace()
        ######
        return response


class ClusterAPI(MethodView):
    def get(self, cluster_id):
        if cluster_id is None:
            # return a list of users
            pass
        else:
            # expose a single user
            pass

    def post(self, cluster_id):
        # create a new cluster
        pass
        ###############
        # print cluster_id
        # pdb.set_trace()
        ###############

    def delete(self, cluster_id):
        # delete a single cluster
        #   -------------------
        print pformat(cluster_id)
        #   -------------------
        try:
            models.ClusterTable.load_and_remove_from_id(cluster_id)
        except Exception as e:
            #################
            # print colored(e)
            print e
            pdb.set_trace()
            #################
            result = {'error': 1}
        else:
            result = {}
        return jsonify(result)

    def put(self, cluster_id):
        # update a single cluster
        req_data = request.get_json()
        #   -------------------
        print pformat(req_data)
        #   -------------------
        updated = {}
        # h5fname = req_data.get('h5fname')
        if 'tags' in req_data:
            result = models.ClusterTable.update_labels(
                labels=req_data['tags'], 
                h5fname = req_data['h5fname'],
                group_path=req_data['group_path'],
                table_name=req_data['table_name'],
            )
            #   Replace 
            # result = dict( (k.replace('/', '-')) for k, v in result.iteritems() )
            updated['tags'] = result
        #   ------------------------------------------------------------
        print 'Updated tags: {0}'.format(colored(pformat(updated), 'magenta'))
        #   ------------------------------------------------------------
        return jsonify(updated)

cluster_view = ClusterAPI.as_view('cluster_api')
app.add_url_rule(
    '/api/clusters', 
    defaults={'cluster_id': None},
    view_func=cluster_view, 
    methods=['GET',]
)
app.add_url_rule(
    '/api/clusters', 
    view_func=cluster_view,
    methods=['POST',]
)
app.add_url_rule(
    '/api/clusters/<cluster_id>', 
    view_func=cluster_view,
    methods=['GET', 'PUT', 'POST', 'DELETE']
)


@app.route('/api/load_raw', methods=['GET', 'POST'])
def load_raw_data():
    #   List files in 'raw_data'
    base_path = os.path.join(THIS_DIR, 'data', 'raw')
    raw_data_files = os.listdir(base_path)
    db_names = {
        'Rock the Ranch'    : 'pm_comments_rtr',
        'All PM Comments'   : 'pm_web_comments'
    }
    raw_data_files.extend(db_names.keys())
    #####
    # pdb.set_trace()
    print raw_data_files
    #####
    req_data = request.get_json()
    if request.method == 'POST':
        #####
        # pdb.set_trace()
        print pformat(req_data)
        #####
        #   Load raw data file
        source_name = req_data['sourcename']
        refresh = req_data['refresh']
        columns = req_data.get('columns', None)
        assert source_name in raw_data_files
        if source_name in db_names:
            df_maker = models.PMClusterDataFrame(
                db_names[source_name], refresh=refresh, columns=columns
            )
        else:
            df_maker = models.ClusterDataFrame(source_name, refresh=refresh, columns=columns)
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
        #####
        # print pformat(children)
        # pdb.set_trace()
        ######
        response = jsonify(*children)
    else:
        #   List raw data files
        result = {'fnames': raw_data_files}
        response = jsonify(**result)
    return response