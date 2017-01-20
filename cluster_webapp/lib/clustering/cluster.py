

"""Contains functionality for transforming a collection of documents into a matrix, performing SVD factorization
and clustering the results.
"""

import os
import itertools
import csv

import simplejson as json

from . import models as cluster_models


#:  The directory containing this script
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def csv_2_json(fname, length=None, doc_type=None):
    """Transforms a CSV file into JSON format.
    """
    with open(fname, mode='rbU') as f:
        reader = csv.DictReader(f)
        rows = ( r for r in reader )
        if length:
            rows = itertools.slice(rows, 0, length)
        if doc_type:
            rows = ( r for r in rows if r['type'] == doc_type )
        rows = list(rows)
    json_fname = os.path.splitext(fname)[0] + '.json'
    ################
    print 'Serializing results to JSON and writing to {0}...'.format(json_fname),
    ################
    try:
        with open(json_fname, mode='wb') as f:
            json.dump(rows, f, indent=' ' * 4)
        print 'Finished!'
    except Exception as e:
        'Failed: {0}'.format(e)



def cluster_by_grouped_topics(dataset_id, num_topics=200, num_clusters=10):
    """Clusters a list of documents."""
    print 'Clustering documents by grouped topics in dataset {0}...'.format(dataset_id)
    clusterer = cluster_models.Clusterer(
        dataset_id=dataset_id,
        num_topics=num_topics,
        num_clusters=num_clusters,
    )
    aggregated_docs = clusterer.cluster_by_grouped_topics()
    return aggregated_docs



def cluster_documents(  dataset_id=None,
                        num_topics=200,
                        num_clusters=10,
                        topics=None,
                        cutoff=.55 ):
    """
    Clusters a list of documents.

        :param dataset_id: The identifier of the dataset to be clustered
        :param num_topics: The number of topics to return (i.e. the number of singular values to retain)
        :param num_clusters: The number of clusters to group documents into
        :param topics: The topics associated with a given Latent Semantic Indexing operation
        :param cutoff: The minimum similarity required for a document to match a topic

    """
    print 'Clustering documents in dataset {0}...'.format(dataset_id)
    clusterer = cluster_models.Clusterer(
        dataset_id=dataset_id,
        num_topics=num_topics,
        num_clusters=num_clusters,
        topics=topics,
        cutoff=cutoff
    )
    relevant_docs_sents = clusterer.cluster()
    return relevant_docs_sents

