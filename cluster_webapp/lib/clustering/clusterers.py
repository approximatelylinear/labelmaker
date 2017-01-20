
"""Contains models used to store clustering data.
"""

#   Stdlib
import os
import time
import csv
import datetime
import collections
import pdb
import logging
from operator import itemgetter
try:
    import cPickle as pickle
except ImportError:
    import pickle
from pprint import pformat

#   3rd party
import simplejson as json
#   Numpy/scipy
import numpy as np
from scipy import linalg
#   Gensim
import gensim
from gensim import models as gensim_models
from gensim import corpora as gensim_corpora
from gensim.similarities.docsim import Similarity
import pandas as pd
from pandas import DataFrame, Series
from termcolor import colored, cprint
#   Scikit-learn
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn import metrics

#   Custom (Use relative imports)
from .exceptions import NoDataError
from .models_document import ClusterDocumentList
from .models_svd import (
    GensimDictionary, GensimCorpus, GensimTFIDFCorpus, GensimLSISpace,
    GensimLSICorpus, GensimSimilarityIndex
)
from . import conf

LOGGER = logging.getLogger(__name__)

#   Matplotlib
try:
    from matplotlib import pyplot as plt
    import pylab as pl
    MATPLOTLIB_IS_AVAILABLE = True
except RuntimeError as exc:
    LOGGER.exception(exc)
    MATPLOTLIB_IS_AVAILABLE = False


def create_doc_mat(dataset_id, from_file=True, df=None):
    if (not from_file) and (df is not None):
        path = ClusterDocumentList(dataset_id).raw_data_path
        df.to_csv(path)
    clusterer = Clusterer(dataset_id, refresh=True)
    ###############################
    msg = 'Creating {0} for dataset {1}...'.format(
        colored('cluster models', 'yellow'),
        colored(str(dataset_id), 'red'),
    )
    LOGGER.info(msg, extra=dict(border=True))
    ###############################
    clusterer.dictionary_wrapper.get_or_create()
    clusterer.corpus_wrapper.get_or_create()
    clusterer.corpus_wrapper.to_csv()
    return clusterer

def create_tfidf_mat(dataset_id, from_file=True, df=None):
    clusterer = create_doc_mat(dataset_id, from_file, df)
    clusterer.tfidf_corpus_wrapper.get_or_create()
    clusterer.tfidf_corpus_wrapper.to_csv()
    return clusterer



class Clusterer(object):
    def __init__(
            self,
            dataset_id,
            df=None,
            num_topics=200,
            num_clusters=10,
            topics=None,
            save=True,
            refresh=False
        ):
        """
        Clusters a list of documents.

            :param dataset_id: The identifier of the dataset to be clustered
            :param num_topics: The number of topics to return (i.e. the number of singular values to retain)
            :param num_clusters: The number of clusters to group documents into
            :param topics: The topics associated with a given Latent Semantic Indexing operation
            :param cutoff: The minimum similarity required for a document to match a topic

        """
        self.dataset_id = dataset_id
        self.num_topics = num_topics
        self.num_clusters = num_clusters
        self.topics = topics
        self.save = save

    @property
    def topics(self):
        try:                    self._topics
        except AttributeError:  self._topics = self.lsi_space_wrapper.get_topics()
        return self._topics

    @topics.setter
    def topics(self, value):
        self._topics = value

    @property
    def topic_sims_sents(self):
        try:
            self._topic_sims_sents
        except AttributeError:
            topic_sims_sents = self.lsi_space_wrapper.transform_topics(self.topics)
            self._topic_sims_sents = list(topic_sims_sents)
        return self._topic_sims_sents

    @property
    def num_clusters(self):
        try:
            self._topic_sims_sents
        except AttributeError:
            pass
        else:
            self._num_clusters = max(self._num_clusters, len(self._topic_sims_sents))
        return self._num_clusters

    @num_clusters.setter
    def num_clusters(self, value):
        self._num_clusters = value

    @property
    def term_list(self):
        try:
            self._term_list
        except AttributeError:
            term_list = Series(self.dictionary)
            term_list = term_list.sort_index()
            self._term_list = term_list
        return self._term_list

    @term_list.setter
    def term_list(self, value):
        self._term_list = value

    def delete_dependencies(self):
        """
        Remove all clustering-related models from the dataset:
            (1) Delete base, tf.idf, and lsi-space corpora
            (2) Delete similarity_indices
            (3) Delete lsi spaces
        """

        self.document_list_wrapper.delete()
        LOGGER.warn('\tDeleted DOCUMENT LIST')
        self.dictionary_wrapper.delete()
        LOGGER.warn('\tDeleted DICTIONARY')
        self.corpus_wrapper.delete()
        LOGGER.warn('\tDeleted BASE CORPUS')
        self.tfidf_corpus_wrapper.delete()
        LOGGER.warn('\tDeleted TFIDF CORPUS')
        self.lsi_space_wrapper.delete()
        LOGGER.warn('\tDeleted LSI PROJECTION')
        self.lsi_space_corpus_wrapper.delete()
        LOGGER.warn('\tDeleted LSI PROJECTION CORPUS')
        self.similarity_index_wrapper.delete()
        LOGGER.warn('\tDeleted SIMILARITY INDEX')

    def create_document_list(self, df):
        """
            Get a list of documents from disk or create one.

        """
        self.document_list_wrapper = ClusterDocumentList(self.dataset_id, df=df)
        self.document_list = self.document_list_wrapper.data

    def create_dictionary(self):
        #   Dictionary
        self.dictionary_wrapper = GensimDictionary(
            self.dataset_id, document_list=self.document_list_wrapper.data,
        )
        self.dictionary = self.dictionary_wrapper.data

    def create_corpus(self):
        #   Standard corpus
        self.corpus_wrapper = GensimCorpus(
            self.dataset_id,
            document_list=self.document_list,
            dictionary=self.dictionary
        )
        self.standard_corpus = self.corpus_wrapper.data

    def create_corpus_tfidf(self):
        #   TF-IDF corpus
        self.tfidf_corpus_wrapper = GensimTFIDFCorpus(
            self.dataset_id,
            document_list=self.document_list,
            dictionary=self.dictionary,
            standard_corpus=self.standard_corpus,
        )
        self.tfidf_corpus = self.tfidf_corpus_wrapper.data

    def create_lsi_space(self):
        #   LSI space
        self.lsi_space_wrapper = GensimLSISpace(
            self.dataset_id, num_topics=self.num_topics,
            dictionary=self.dictionary,
            tfidf_corpus=self.tfidf_corpus,
        )
        self.lsi_space = self.lsi_space_wrapper.data

    def create_corpus_lsi(self):
        #   LSI-space corpus
        self.lsi_space_corpus_wrapper = GensimLSICorpus(
            self.dataset_id, num_topics=self.num_topics,
            tfidf_corpus=self.tfidf_corpus,
            lsi_space=self.lsi_space,
        )
        self.lsi_space_corpus = self.lsi_space_corpus_wrapper.data

    def create_similarity_index(self):
        #   Similarity index
        self.similarity_index_wrapper = GensimSimilarityIndex(
            self.dataset_id, num_topics=self.num_topics,
            lsi_space_corpus=self.lsi_space_corpus,
        )
        self.similarity_index = self.similarity_index_wrapper.data

    def create_models(self, df=None, delete_dependencies=False):
        """
            Create necessary data structures for clustering a dataframe.

            :param DataFrame df: DataFrame
            :param bool delete_dependencies: Whether to delete model dependencies

            (1) Create a ClusterDocumentList (self.document_list_wrapper)
            (2) Create a GensimDictionary (self.dictionary_wrapper)
            (3) Create a GensimCorpus (self.corpus_wrapper)
            (4) Create a GensimTFIDFCorpus (self.tfidf_corpus_wrapper)
            (5) Create a GensimLSISpace (self.lsi_space_wrapper)
            (6) Create a GensimLSICorpus (self.lsi_space_corpus_wrapper)
            (7) Create a GensimSimilarityIndex (self.similarity_index_wrapper)

        """
        ###############################
        msg = 'Creating {0} for dataset {1}...'.format(
            colored('cluster models', 'yellow'),
            colored(str(self.dataset_id), 'red'),
        )
        LOGGER.info(msg, extra=dict(border=True))
        ###############################
        self.create_document_list(df)
        self.create_dictionary()
        self.create_corpus()
        self.create_corpus_tfidf()
        self.create_lsi_space()
        self.create_corpus_lsi()
        self.create_similarity_index()
        ###############################
        msg = 'Done!'
        LOGGER.info(msg)
        ###############################

    def cluster(self):
        raise NotImplementedError

    def term_cluster(self):
        #   ----------------------------------------------
        msg = 'Clustering terms in dataset {0}...'.format(
            colored(self.dataset_id, 'yellow')
        )
        LOGGER.info(msg, extra=dict(border=True))
        #   ----------------------------------------------
        try:
            self.create_models()
            #   Get rank-k matrix of *left* singular vectors of A
            #       Columns of U[:, :k]
            #   Convert U to a sparse matrix ...
            U = lsi_space_wrapper.U
            labels = self.cluster(U)
            #   reorder rows of A by indices in sorted vector x
            term_list = DataFrame(self.term_list, columns=['term'])
            term_list['cluster_id'] = labels
            term_list = term_list.sort_index(by='cluster_id')
        except NoDataError:
            #   ----------------------------------------
            LOGGER.error('No data to cluster.')
            #   ----------------------------------------
            term_list = DataFrame()
        self.term_list = term_list
        return term_list

    def doc_cluster(self, df=None):
        msg = 'Clustering documents in dataset {}...'.format(
            colored(self.dataset_id), 'yellow'
        )
        LOGGER.info(msg, extra=dict(border=True))
        try:
            topics = self.lsi_space_corpus_wrapper.get_topics(5)
            #   -------------------------------------------------
            msg = '\nTop 5 topics in the dataset'
            """
            for t in topics: t_ = ', '.join(['*'.join([colored(t_['v'], 'red'), colored(t_['w'], 'cyan')]) for t_ in t['t']]); print '\t[{0}]\t{1}'.format(colored(str(t['i']), 'yellow'), t_)
            """
            for t in topics:
                t_ = ', '.join(
                    ['*'.join([colored(t_['v'], 'red'), colored(t_['w'], 'cyan')]) for t_ in t['t']]
                )
                msg += '\n\t[{0}]\t{1}'.format(colored(str(t['i']), 'yellow'), t_)
            LOGGER.info(msg, extra=dict(border=True))
            #   -------------------------------------------------

            #   Get rank-k matrix of *right* singular vectors of A
            #       Columns of V[:, :k] (i.e. rows of V'[:k, :]
            V = self.lsi_space_corpus_wrapper.V #   Need the corpus before getting V.
            data_df = self.cluster(V, plot=False)
            data_df = data_df.rename(columns={'labels': 'cluster_id'})
            doc_list = self.document_list
            doc_list = pd.merge(data_df, doc_list, left_index=True, right_index=True, suffixes=['', '_del'])
            doc_list = doc_list.drop(['cluster_id_del'], axis=1)
            #   reorder rows of A by indices in sorted vector of labels.
            doc_list = doc_list.sort_index(by='cluster_id')
        except NoDataError:
            #   ----------------------------------------
            msg = 'No data to cluster.'
            LOGGER.error(msg)
            #   ----------------------------------------
            doc_list = DataFrame()
        except Exception as e:
            #   --------------------------------------------------
            msg = 'Error clustering: {}'.format(colored(e, 'red'))
            LOGGER.error(msg)
            doc_list = DataFrame()
            #   --------------------------------------------------
        self.doc_list = doc_list
        return doc_list

    def get_info_features(self):
        try:
            corpus = self.tfidf_corpus_wrapper.data
            corpus = gensim.matutils.corpus2csc(corpus)
            np_corpus = corpus.toarray()
            avg_counts = np_corpus / self.num_clusters
            avg_counts = np.sum(avg_counts, axis=1)
            #   Normalize counts.
            avg_counts = avg_counts / np_corpus.shape[1]
            avg_counts = Series(avg_counts)
            word_ids = Series(self.dictionary_wrapper.data.token2id)
            doc_list = self.doc_list
            groups = doc_list.groupby('cluster_id', group_keys=False)
            #   Get the divergence for each item in the current cluster from the average

            def calc_cluster_diff(group):
                try:
                    #   Grab the documents corresponding to the indexes in the document ids.
                    ids = group['docnum']
                    #   The np_corpus is ordered (word_ids, doc_ids)
                    curr_docs = np_corpus[:, ids]
                    curr_counts = np.sum(curr_docs, axis=1)
                    #   Normalize counts.
                    curr_counts = curr_counts / curr_docs.shape[1]
                    diff_counts = Series(curr_counts - avg_counts)
                    #   Construct a dataframe out of the counts
                    #   Swap the index/values for the word ids, so that they're indexed by row num.
                    _word_ids = Series(word_ids.index, word_ids.values)
                    counts = DataFrame({'counts': diff_counts, 'ids': _word_ids})
                    cluster_id = group.cluster_id.values[0]
                    counts['cluster_id'] = cluster_id
                    counts = counts.dropna()
                    counts = counts[counts.counts > 0]
                    counts = counts.sort_index(by='counts')[::-1][:20]
                    counts.index = xrange(len(counts))
                except Exception as e:
                    LOGGER.error(e)
                    raise
                return counts
            all_counts = groups.apply(calc_cluster_diff)
        except Exception as e:
            LOGGER.error(e)
            raise
        return all_counts



class SignClusterer(Clusterer):
    def __init__(
            self,
            dataset_id,
            num_topics=200,
            num_clusters=3,
            topics=None,
            save=True,
            refresh=False
        ):
        super(SignClusterer, self).__init__(
            dataset_id=dataset_id,
            num_topics=num_topics,
            num_clusters=num_clusters,
            topics=topics,
            save=save,
            refresh=refresh
        )

    def cluster(self, data, plot=False):
        """
        http://meyer.math.ncsu.edu/Meyer/Talks/ClusteringMonopoli_9_11_07.pdf
        """
        #   ------------------------------------------
        msg = colored('\tNow clustering...', 'yellow')
        LOGGER.info(msg, extra=dict(border=True))
        #   ------------------------------------------
        x, data_j = self.get_sign_pattern(data)
        unique_labels = np.unique(x)
        labels = Series(dict(zip(xrange(len(x)), x)))
        #   ----------------------------------------------------
        msg = '\t\t{msg}: {num} (target maximum: {tnum})'.format(
            msg=colored('Number of clusters found', 'yellow'),
            num=colored(str(len(unique_labels)), 'red'),
            tnum=colored(str(2 ** self.num_clusters), 'red')
        )
        LOGGER.info(msg)
        #   ----------------------------------------------------
        ########
        # print x
        # print data_j
        ########
        data_df = DataFrame(
            data_j, columns=['v' + str(i) for i in xrange(data_j.shape[1])]
        )
        data_df['labels'] = labels
        if plot: self.plot_clusters(data)
        return data_df

    def get_sign_pattern(self, M):
        #   ------------------------------------------
        msg = colored('\t\tGetting sign patterns...', 'yellow')
        LOGGER.info(msg, extra=dict(border=True))
        #   ------------------------------------------
        #   M is actually VS, but that's okay because:
        #       (a) We only want the sign pattern.
        #       (b) The S entries, being square-roots, are all non-negative.
        n, m = M.shape
        j = min(self.num_clusters, m)
        #   Create binary matrix with sign patterns
        #       B = V[:, 1:j]
        B = M[:, 0:j]
        B_pos = np.where(B >= 0, 1, 0)
        x = np.zeros(n)
        for i in xrange(j):
            #   Convert binary repr. to base 10.
            s = 2 ** (j - i)
            B_pos_i = B_pos[:, i]
            x = np.add(x, (s * B_pos_i))
        return x, B

    def plot_clusters(self, data):
        raise NotImplementedError



class SpectralClusterer(Clusterer):
    def __init__(
            self,
            dataset_id,
            num_topics=200,
            num_clusters=3,
            topics=None,
            save=True,
            refresh=False
        ):
        super(SpectralClusterer, self).__init__(
            dataset_id=dataset_id,
            num_topics=num_topics,
            num_clusters=num_clusters,
            topics=topics,
            save=save,
            refresh=refresh
        )

    def cluster(self, data, plot=False, labels=None):
        norm_data = np.exp(-data / data.std())
        sp_model = SpectralClustering(n_clusters=self.num_clusters, mode='arpack')
        labels = sp_model.fit_predict(data)
        #   -------------------------------
        msg = 'Fitting the Spectral Clustering model...'
        LOGGER.info(msg, extra=dict(border=True))
        #   -------------------------------
        unique_labels = np.unique(labels)
        labels = Series(dict(zip(xrange(len(labels)), labels)))
        #   ------------------------------------------------
        msg = '\t\t{msg}: {num} (target maximum: {tnum})'.format(
            msg=colored('Number of clusters found', 'yellow'),
            num=colored(str(len(unique_labels)), 'red'),
            tnum=colored(str(self.num_clusters), 'red')
        )
        LOGGER.info(msg)
        #   ------------------------------------------------
        if plot: self.plot_clusters(data)
        return labels



class DBSCANClusterer(Clusterer):
    def __init__(
            self,
            dataset_id,
            num_topics=200,
            num_clusters=3,
            topics=None,
            save=True,
            refresh=False
        ):
        super(SpectralClusterer, self).__init__(
            dataset_id=dataset_id,
            num_topics=num_topics,
            num_clusters=num_clusters,
            topics=topics,
            save=save,
            refresh=refresh
        )

    def cluster(self, data, plot=False, labels=None):
        norm_data = np.exp(-data / data.std())
        db_model = DBSCAN(eps=0.95, min_samples=10)
        labels = db_model.fit_predict(data)
        #   -------------------------------
        msg = 'Fitting the DBSCAN Clustering model...'
        LOGGER.info(msg, extra=dict(border=True))
        #   -------------------------------
        unique_labels = np.unique(labels)
        labels = Series(dict(zip(xrange(len(labels)), labels)))
        #   ------------------------------------------------
        msg = '\t\t{msg}: {num} (target maximum: {tnum})'.format(
            msg=colored('Number of clusters found', 'yellow'),
            num=colored(str(len(unique_labels)), 'red'),
            tnum=colored(str(self.num_clusters), 'red')
        )
        LOGGER.info(msg)
        #   ------------------------------------------------
        if plot: self.plot_clusters(data)
        return labels



class KMeansClusterer(Clusterer):
    def __init__(
            self,
            dataset_id,
            num_topics=200,
            num_clusters=3,
            topics=None,
            save=True,
            refresh=False
        ):
        super(KMeansClusterer, self).__init__(
            dataset_id=dataset_id,
            num_topics=num_topics,
            num_clusters=num_clusters,
            topics=topics,
            save=save,
            refresh=refresh
        )

    @staticmethod
    def bench_k_means(estimator, name, data, n_samples=None, labels=None):
        t0 = time.time()
        estimator.fit(data)
        sil = metrics.silhouette_score(
            data,
            estimator.labels_,
            metric='euclidean',
#             sample_size=n_samples
        )
        base_msg = '{itia:.3f}\t{sil:.3f}'.format(
            nm=name,
            tm=(time.time() - t0),
            itia=estimator.inertia_,
            sil=sil
        )
        lbl_msg = ''
        if labels:
            lbl_msg = '{hm:.3f}\t{cm:.3f}\t{vm:.3f}\t{rs:.3f}\{mis:.3f}'.format(
                hm=metrics.homogeneity_score(labels, estimator.labels_),
                cm=metrics.completeness_score(labels, estimator.labels_),
                vm=metrics.v_measure_score(labels, estimator.labels_),
                rs=metrics.adjusted_rand_score(labels, estimator.labels_),
                mis=metrics.adjusted_mutual_info_score(labels, estimator.labels_),
            )
        msg = base_msg + lbl_msg + '\n'
        LOGGER.info(msg, extra=dict(border=True))

    def cluster(self, data, plot=False, labels=None):
        n_samples, n_features = data.shape
        n_init = 20 #10
        init = 'random' #   'k-means++'
        km_model = KMeans(init=init, n_clusters=self.num_clusters, n_init=n_init)
        #   -------------------------------
        msg = 'Fitting the K-Means model...'
        LOGGER.info(msg, extra=dict(border=True))
        #   -------------------------------
        km_model.fit(data)
        labels = km_model.labels_
        centroids = km_model.cluster_centers_
        unique_labels = np.unique(labels)
        labels = Series(dict(zip(xrange(len(labels)), labels)))
        #   ------------------------------------------------
        msg = '\t\t{msg}: {num} (target maximum: {tnum})'.format(
            msg=colored('Number of clusters found', 'yellow'),
            num=colored(str(len(unique_labels)), 'red'),
            tnum=colored(str(self.num_clusters), 'red')
        )
        LOGGER.info(msg)
        #   ------------------------------------------------
        if plot: self.plot_clusters(data)
        return labels

    def plot_clusters(self, data):
        if not MATPLOTLIB_IS_AVAILABLE:
            raise Exception("Cannot plot clusters: matplotlib is not installed.")
        reduced_data = PCA(n_components=2).fit_transform(data)
        reduced_data = data[:, 0:2]
        r_km_model = KMeans(init='k-means++', n_clusters=self.num_clusters, n_init=10)
        r_km_model.fit(reduced_data)
        # Step size of the mesh. Decrease to increase the quality of the VQ.
        h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].
        # Plot the decision boundary. For that, we will asign a color to each
        x_min, x_max = reduced_data[:, 0].min() + 1, reduced_data[:, 0].max() - 1
        y_min, y_max = reduced_data[:, 1].min() + 1, reduced_data[:, 1].max() - 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # Obtain labels for each point in mesh. Use last trained model.
        Z = r_km_model.predict(np.c_[xx.ravel(), yy.ravel()])
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        pl.figure(1)
        pl.clf()
        pl.imshow(
            Z,
            interpolation='nearest',
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap=pl.cm.Paired,
            aspect='auto',
            origin='lower'
        )
        pl.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
        # Plot the centroids as a white X
        centroids = r_km_model.cluster_centers_
        pl.scatter(
            centroids[:, 0],
            centroids[:, 1],
            marker='x',
            s=169,
            linewidths=3,
            color='w',
            zorder=10
        )
        pl.title(
            'K-means clustering on PCA-reduced data\n'
            'Centroids are marked with white cross'
        )
        pl.xlim(x_min, x_max)
        pl.ylim(y_min, y_max)
        pl.xticks(())
        pl.yticks(())
        pl.show()
