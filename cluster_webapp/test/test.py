
import logging
import pdb
from pprint import pformat

import conf
import models, models_metadata

def test_load_file():
    source_name = 'example.csv'
    df_maker = models.ClusterDataFrame(
        source_name, refresh=True, columns=None
    )
    df = df_maker.load()
    # children = df_maker.to_h5()
    return df, df_maker

def test_serialize():
    #   Serialize
    #   Unserialize
    pass


def test_load_db(limit=10):
    source_name = 'PM UGC'
    session = models_metadata.SESSION()
    raw_data_files = models_metadata.list_datasets()
    names_2_pk = dict(raw_data_files)
    pk = names_2_pk[source_name]
    obj = session.query(models_metadata.Dataset).get(pk) if pk else None
    db_name, df_cls = obj.name_db, obj.manager
    df_cls = getattr(models, df_cls.split('.')[-1])
    df_maker = df_cls(db_name, refresh=False,)
    db_info = df_maker.get_info()
    print db_info
    filters = {}
    filters['limit'] = limit
    df_maker = df_cls(
        db_name,
        refresh=False,
        filters=filters,
        fields=None
    )
    df = df_maker.load()
    ##########
    pdb.set_trace()
    ##########
    return df, df_maker


def cluster(req_data):
    table_name = req_data['table_name']         #   table name
    group_path = req_data['group_path']         #   h5 path
    h5fname = req_data['h5fname']                 #   OS location of h5 db
    cluster_method = req_data.get('cluster_method', None) #   Name of clustering method
    db_name = req_data['db_name'] or None
    #   Retrieve cluster from h5 store.
    cluster_maker = models.Cluster(
        table_name=table_name,
        group_path=group_path,
        num_clusters=1,
        refresh=False,
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
    return result


def test_cluster():
    # source_name = 'example.csv'
    # df_maker = models.ClusterDataFrame(
    #     source_name, refresh=True, columns=None
    # )
    # df, df_maker = test_load_db(limit=100)
    df, df_maker = test_load_file()
    children = df_maker.to_h5()
    req_data = children[0]
    clusters = cluster(req_data)
    print pformat(clusters)
    pdb.set_trace()



if __name__ == '__main__':
    config = conf.setup_logging()
    logging.config.dictConfig(config)
    # test_load_file()
    # test_load_db()
    test_cluster()
