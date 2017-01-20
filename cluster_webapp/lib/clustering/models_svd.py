
"""
Models for performing Singular Value Decomposition on a document matrix with Gensim
"""

#   Stdlib
import os
import datetime
import logging
import pdb
try:
    import cPickle as pickle
except ImportError:
    import pickle

#   3rd party
from termcolor import colored
#       Numpy/scipy
from scipy import linalg
#       Gensim
import gensim
from gensim import models as gensim_models
from gensim.similarities.docsim import Similarity

#   Custom current
from .util import mproperty
from .exceptions import NoDataError
from .models_document import GensimDictionary, GensimCorpus, GensimTFIDFCorpus
from . import conf

LOGGER = logging.getLogger(__name__)


class GensimLSISpace(object):
    """Stores a reference to the file containing a gensim lsi (SVD matrix) object.

    NB (from the docs):
        The left singular vectors are saved in lsi.projection.u,
        singular values in lsi.projection.s.
        Right singular vectors can be reconstructed from the output of lsi[training_corpus]
    """
    lsi_space_choices = (('1', 'Standard'), )
    STANDARD = '1'
    category = 1

    def __init__(self,
                 dataset_id,
                 num_topics=200,
                 dictionary=None,
                 tfidf_corpus=None, ):
        self.dataset_id = dataset_id
        self.num_topics = num_topics
        if dictionary is not None: self.dictionary = dictionary
        if tfidf_corpus is not None: self.tfidf_corpus = tfidf_corpus
        self.loaded = False

    @mproperty
    def data(self):
        return self.fit_transform()

    @mproperty
    def S(self):
        if self.data is not None:
            S = self.data.projection.s
        else:
            raise RuntimeError(
                'Must create the LSI space before retrieving S.')
        return S

    @mproperty
    def U(self):
        if self.data is not None:
            U = self.data.projection.u
        else:
            raise RuntimeError('Must create the LSI space before retrieving U.')
        return U

    @mproperty
    def path(self):
        """Creates a path for storing and accessing LSI space objects."""
        base_path = conf.PATH_LSI_SPACES
        fname = '_'.join([str(self.dataset_id), 'lsi_spaces', str(self.category)])
        fname = '{}.lsi'.format(fname)
        return os.path.join(base_path, fname)

    @mproperty
    def tfidf_corpus(self):
        tfidf_corpus_wrapper = GensimTFIDFCorpus(
            self.dataset_id,
            num_topics=self.num_topics,
            dictionary=self.dictionary)
        return tfidf_corpus_wrapper.fit_transform()

    @tfidf_corpus.setter
    def tfidf_corpus(self, value):
        self._tfidf_corpus = value

    @mproperty
    def dictionary(self):
        dictionary_wrapper = GensimDictionary(self.dataset_id)
        return dictionary_wrapper.fit_transform()

    @dictionary.setter
    def dictionary(self, value):
        self._dictionary = value

    def fit_transform(self):
        """Gets or creates an LSI space object."""
        try:
            lsi_space = self.get()
        except Exception as e:
            lsi_space = self._fit_transform()
        return lsi_space

    def get(self):
        """Gets an LSI space object."""
        path = self.path
        lsi_space = self.load(path)
        #   -----------------------------------
        msg = 'Retrieved an existing {} transformation.'.format(colored(
            'LSI space', 'yellow'))
        LOGGER.info(msg)
        #   -----------------------------------
        return lsi_space

    def _fit_transform(self):
        """Projects a corpus onto LSI space."""
        #   -----------------------------------
        msg = 'Projecting a corpus onto LSI space...'
        LOGGER.info(msg, extra=dict(border=True))
        #   -----------------------------------
        tfidf_corpus = self.tfidf_corpus
        dictionary = self.dictionary
        lsi_space = gensim_models.LsiModel(tfidf_corpus,
                                           id2word=dictionary,
                                           num_topics=self.num_topics)
        self.save(lsi_space)
        return lsi_space

    def delete(self):
        """Deletes an LSI space object."""
        path = self.path
        if os.path.exists(path):
            os.remove(path)

    def load(self, path):
        """Loads an existing LSI space object."""
        lsi_space = gensim_models.LsiModel.load(path)
        return lsi_space

    def save(self, lsi_space):
        """Pickles a gensim LSI space object.
        """
        #   Use the Gensim method to save it.
        lsi_space.save(self.path)

    def get_topics(self, n=5):
        """Returns the topics associated with a particular Latent Semantic Indexing
        operation.

        """
        if self.data is not None:
            topics = self.data.obj.show_topics(num_topics=5, formatted=False)
            topics = [[{'i': i, 't': [{'w': w, 'v': v} for w, v in t]}] for i, t in topics]
        else:
            raise NoDataError
        return topics

    def transform_topics(self, topics):
        """Projects a given set of topics into the space defined by
        a Singular Value Decomposition.
        """
        topics = []
        if self.data and self.dictionary:
            topics = ((self.dictionary.doc2bow(topic), topic)
                      for topic in topics)
            topics = ((self.data[sim], sent) for sim, sent in topics)
        return topics


class GensimLSICorpus(GensimCorpus):
    category = 3

    def __init__(self, *args, **kwargs):
        tfidf_corpus = kwargs.pop('tfidf_corpus', None)
        if tfidf_corpus is not None: self.tfidf_corpus = tfidf_corpus
        lsi_space = kwargs.pop('lsi_space', None)
        if lsi_space is not None: self.lsi_space = lsi_space
        super(GensimLSICorpus, self).__init__(*args, **kwargs)

    @mproperty
    def VS(self):
        """
            Get rank-k matrix of *right* singular vectors of A * S
            Columns of VS[:, :k] (i.e. rows of SV'[:k, :]
        """
        if self.data is not None:
            #   -----------------------------------------------------------
            msg = colored('\t\tGetting right singular vectors (VS)...',
                          'yellow')
            LOGGER.info(msg)
            #   -----------------------------------------------------------
            #   self.lsi_space[self.tfidf_corpus] = U'X
            #       U'X = SV'
            #       V' = S^-1(U'X)
            #       V = (S^-1(U'X))'
            _ = self.lsi_space  #   Make sure the object exists
            S = self.lsi_space_wrapper.S
            #   Sparse version
            SVt = gensim.matutils.corpus2csc(self.data, len(S))
            VS = SVt.T
        else:
            raise RuntimeError(
                'Must create the LSI space corpus before retrieving VS.')
        return VS

    @mproperty
    def V(self):
        """
            Get rank-k matrix of *right* singular vectors of A
            Columns of V[:, :k] (i.e. rows of V'[:k, :]
        """
        if self.data is not None:
            #   -----------------------------------------------------------
            msg = colored('\t\tGetting right singular vectors (V)...',
                          'yellow')
            LOGGER.info(msg)
            #   -----------------------------------------------------------
            #   self.lsi_space[self.tfidf_corpus] = U'X
            #       U'X = SV'
            #       V' = S^-1(U'X)
            #       V = (S^-1(U'X))'
            _ = self.lsi_space  #   Make sure the object exists
            S = self.lsi_space_wrapper.S
            #   Dense version
            UtX = gensim.matutils.corpus2dense(self.data, len(S))
            #   Convert 1d array to S x S matrix.
            S = linalg.diagsvd(S, len(S), len(S))
            Vt = linalg.solve(S, UtX)  #   ==  np.inv(S).dot(UtX)
            _V = Vt.T
        else:
            raise RuntimeError(
                'Must create the LSI space corpus before retrieving V.')
        return _V

    @mproperty
    def tfidf_corpus_wrapper(self):
        return GensimTFIDFCorpus(
            self.dataset_id,
            num_topics=self.num_topics,
            document_list=self.document_list,
            dictionary=self.dictionary)

    @mproperty
    def tfidf_corpus(self):
        return self.tfidf_corpus_wrapper.fit_transform()

    @tfidf_corpus.setter
    def tfidf_corpus(self, value):
        self._tfidf_corpus = value

    @mproperty
    def lsi_space_wrapper(self):
        return GensimLSISpace(self.dataset_id, num_topics=self.num_topics)

    @mproperty
    def lsi_space(self):
        return self.lsi_space_wrapper.fit_transform()

    @lsi_space.setter
    def lsi_space(self, value):
        self._lsi_space = value

    def get(self):
        """Gets an lsi space corpus."""
        path = self.path
        corpus = self.load(path)
        #   -----------------------------------
        msg = 'Found an existing {} with {1} entries'.format(
            colored('LSI space corpus', 'yellow'), colored(
                str(len(corpus)), 'red'))
        LOGGER.info(msg)
        #   -----------------------------------
        return corpus

    def _fit_transform(self):
        """Projects values in a TF-IDF corpus onto LSI (i.e. SVD) coordinates
        using gensim routines.
        """
        #   -----------------------------------
        msg = 'Creating an {}...'.format(colored('LSI space corpus',
                                                  'yellow'))
        LOGGER.info(msg, extra=dict(border=True))
        #   -----------------------------------
        #   Project the tf.idf corpus onto LSI space
        #       self.lsi_space[self.tfidf_corpus] = U^TX
        lsi_space_corpus = self.lsi_space[self.tfidf_corpus]
        self.save(lsi_space_corpus)
        return lsi_space_corpus

    def get_topics(self, n=5):
        """Returns the topics associated with a particular Latent Semantic Indexing
        operation.

        """
        if self.data is not None:
            topics = self.data.obj.show_topics(num_topics=5, formatted=False)
            topics = [{'i': i, 't': [{'w': w, 'v': v} for w, v in t]} for i, t in topics]
        else:
            raise NoDataError
        return topics

class GensimSimilarityIndex(object):
    """Stores a reference to the file containing a gensim similarity index.
    """
    similarity_index_choices = (('1', 'Standard'), )
    STANDARD = '1'
    category = 1

    def __init__(self, dataset_id, num_topics=200, lsi_space_corpus=None):
        self.dataset_id = dataset_id
        self.num_topics = num_topics
        if lsi_space_corpus is not None:
            self.lsi_space_corpus = lsi_space_corpus

    @mproperty
    def data(self):
        return self.fit_transform()

    @mproperty
    def path(self):
        """Creates a path for storing and accessing similarity indices."""
        base_path = conf.PATH_SIMILARITY_INDICES
        fname = '_'.join([str(self.dataset_id), 'similarity_indices', str(
            self.category)])
        fname = '{}.simindex'.format(fname)
        return os.path.join(base_path, fname)

    @mproperty
    def lsi_space_corpus(self):
        lsi_corpus_wrapper = GensimLSICorpus(self.dataset_id,
                                             num_topics=self.num_topics)
        return lsi_corpus_wrapper.fit_transform()

    @lsi_space_corpus.setter
    def lsi_space_corpus(self, value):
        self._lsi_space_corpus = value

    def fit_transform(self):
        """Gets or creates a similarity index."""
        try:
            similarity_index = self.get()
        except Exception as e:
            similarity_index = self._fit_transform()
        return similarity_index

    def get(self):
        """Gets a similarity index."""
        path = self.path
        similarity_index = self.load(path)
        #   -----------------------------------
        msg = 'Retrieved a similarity index.'
        LOGGER.info(msg)
        #   -----------------------------------
        return similarity_index

    def _fit_transform(self):
        """Gets the (previously computed) LSI space transform of a corpus, and
        then calculates the similarities among the entries.
        """
        #   -----------------------------------
        msg = 'Creating a {}...'.format(colored('similarity index', 'yellow'))
        LOGGER.info(msg, extra=dict(border=True))
        #   -----------------------------------
        lsi_space_corpus = self.lsi_space_corpus
        ##  Permit Gensim to break apart matrices across separate files
        ##  Set `num_features` equal to the number of lsi dimensions.
        now = datetime.datetime.now().strftime('%c').replace(
            ' ', '_').replace(':', '_')
        output_path = os.path.join(conf.PATH_RAW_SIMILARITY_INDICES, 'index_{}'.format(now))
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        similarity_index = Similarity(output_prefix=output_path,
                                      corpus=lsi_space_corpus,
                                      num_features=self.num_topics)
        self.save(similarity_index)
        return similarity_index

    def delete(self):
        """Deletes a similarity index."""
        path = self.path
        if os.path.exists(path): os.remove(path)

    def load(self, path):
        """Loads an existing similarity index."""
        with open(path, mode='rb') as f:
            similarity_index = pickle.load(f)
            return similarity_index

    def save(self, similarity_index):
        """Pickles a gensim similarity index.
        """
        path = self.path
        with open(path, 'wb') as f:
            pickle.dump(similarity_index, f)
