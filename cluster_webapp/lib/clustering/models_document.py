
#   Stdlib
import os
import logging
try:
    import cPickle as pickle
except ImportError:
    import pickle

#   3rd party
from termcolor import colored
import pandas as pd
from pandas import DataFrame, Series
#       Gensim
import gensim
from gensim import models as gensim_models
from gensim import corpora as gensim_corpora


#   Custom external
# from textclean.spacy_processor import SpacyProcessor
from textclean.v1.util import STOPWORDS
from textclean.v1.util import Pipeline

#   Custom current
from .util import mproperty
from .exceptions import NoDataError
from . import conf

LOGGER = logging.getLogger(__name__)

# Spacy processor
# SPACY_PROCESSOR = SpacyProcessor()
# SPACY_PROCESSOR = None

class ClusterDocumentList(object):
    """
    A list of all clusters and documents associated with a dataset
    """

    def __init__(self, dataset_id, df=None):
        self._id = ''
        #   Foreign Key to a Dataset
        self.dataset_id = dataset_id
        self.df_input = df

    @mproperty
    def data(self):
        """
        DataFrame of data in the cluster. (Cached in `self._data`)
        """
        return self.fit_transform(df=self.df_input)

    @mproperty
    def path(self):
        """Creates a path for accessing this model's data."""
        base_path = conf.PATH_CLUSTER_DOCUMENT_LISTS
        fname = '{}.pkl'.format(self.dataset_id)
        return os.path.join(base_path, fname)

    @mproperty
    def raw_data_path(self):
        """Creates a path for accessing raw data for building this model."""
        base_path = conf.PATH_CLUSTER_DOCUMENTS
        fname = '{}.pkl'.format(self.dataset_id)
        return os.path.join(base_path, fname)

    @mproperty
    def doc_maker(self):
        return ClusterDocumentMaker()

    def fit_transform(self, df=None):
        """Gets or creates a document list."""
        path = self.path
        try:
            if os.path.exists(path):
                documents = self.get()
            else:
                #   Create a new document list
                documents = self._fit_transform(documents=df)
        except Exception as e:
            LOGGER.exception(e)
        return documents

    def get(self):
        """Gets an existing document list."""
        path = self.path
        df = pd.read_pickle(path)
        if isinstance(df, dict):
            df = DataFrame(df)
        #  ------------------------------------------------------
        msg = 'Found an existing {} with {} entries'.format(
            colored('document list', 'yellow'), colored(
                str(len(df)), 'red'))
        LOGGER.info(msg)
        #  ------------------------------------------------------
        return df

    def _fit_transform(self, documents=None):
        """Creates a new document list."""
        #  ------------------------------------------------------
        msg = 'Creating a new {}...'.format(colored('cluster document list',
                                                     'yellow'))
        LOGGER.info(msg)
        try:
            if documents is None:
                documents = pd.read_pickle(self.raw_data_path)
            if 'clean_tokens' in documents:
                if any(documents.clean_tokens.map(
                        lambda x: isinstance(x, basestring))):
                    try:
                        documents['clean_tokens'] = documents[
                            'clean_tokens'].fillna('')
                    except ValueError:
                        documents['clean_tokens'] = ''
                    documents['clean_tokens'] = documents['clean_tokens'].map(
                        lambda x: x.split('|'))
        except Exception as e:
            LOGGER.error(e)
        try:
            cols = documents.columns
            documents = (dict(zip(cols, v)) for v in documents.values)
            doc_maker = self.doc_maker
            cluster_documents = (doc_maker(doc) for doc in documents)
            cluster_documents = (doc for doc in cluster_documents if doc)
            cluster_documents = list(cluster_documents)
            cluster_documents = DataFrame(cluster_documents)
            #   Reset the document counter
            self.doc_maker.doc_num = 0
            if (cluster_documents is None) or cluster_documents.empty:
                raise NoDataError
            #   Save the document list.
            self.save(cluster_documents)
            #################
            #   Check that the save was successful
            df2 = self.get()
            assert df2.equals(cluster_documents)
            #################
        except Exception as e:
            LOGGER.exception(e)
        return cluster_documents

    def save(self, df):
        """Saves a document list."""
        path = self.path
        if (df is None) or df.empty: raise NoDataError
        df_ = df.to_dict()
        with open(path, 'wb') as f:
            pickle.dump(df_, f)

    def delete(self):
        """Deletes a document list."""
        path = self.path
        if os.path.exists(path): os.remove(path)



def make_text_processing_analyzer(**kwargs):
    from sklearn.feature_extraction.text import CountVectorizer
    from textclean.v1 import (StandardizeText, NormalizeSpecialChars,
                          NormalizeWordLength, RemoveStopwords,
                          TokenizeToWords)
    kwargs.setdefault('min_df', 2)
    preprocess_pipeline = (
        StandardizeText(is_html=False,
                        unicode_how=['english_symbol', 'nonascii'])
        | NormalizeSpecialChars() | NormalizeWordLength(
            how=['boundaries', 'nonwhitespace', 'whitespace']))
    tokenize_pipeline = (TokenizeToWords()
                         | RemoveStopwords(out_format='list'))
    vectorizer = CountVectorizer(preprocessor=preprocess_pipeline,
                                 tokenizer=tokenize_pipeline)
    analyzer = vectorizer.build_analyzer()
    return analyzer


# def make_spacy_analyzer(**kwargs):
#     def _analyzer(docs):
#         docs = SPACY_PROCESSOR.parse(docs)
#         docs = SPACY_PROCESSOR.fmt(docs)
#         docs = SPACY_PROCESSOR.remove_spaces(docs)
#         docs = SPACY_PROCESSOR.remove_punct(docs)
#         docs = SPACY_PROCESSOR.remove_stops(docs)
#         docs = SPACY_PROCESSOR.remove_high_freq(docs, probs=SPACY_PROCESSOR.probs)
#         docs = SPACY_PROCESSOR.fmt_tokens(docs)
#         docs = ( doc['tokens'] for doc in docs )
#         return docs

#     return _analyzer


class ClusterDocumentMaker(Pipeline):
    """
    A document in a format appropriate for the clustering routines
    """

    def __init__(self, analyzer=None, *args, **kwargs):
        super(ClusterDocumentMaker, self).__init__(*args, **kwargs)
        self.docnum = 0
        if analyzer is None:
            #   Default to the text_processing_analyzer with the standard keyword args.
            analyzer = make_text_processing_analyzer()
        self.analyzer = analyzer

    def process_item(self, doc, **kwargs):
        """Creates a new cluster document.

        (TBD: )
        Document format:

        {
            '__doc': <original document>,
            '__meta': {
                'error': <error info if any>,
                'text': <text assembled from document components>
            }
        }
        """
        try:
            doc['docnum'] = self.docnum
            _id = doc.get('_id', '')
            clean_tokens = doc.get('clean_tokens', [])
            if not clean_tokens:
                #   -----------
                LOGGER.info('analyzing...')
                #   -----------
                #   These should have already been cleaned.
                simplified_text = doc['__meta']['text']
                #   -----------------------------------
                LOGGER.debug(simplified_text)
                #   -----------------------------------
                clean_tokens = self.analyzer(simplified_text)
                doc['clean_content'] = simplified_text
                doc['clean_tokens'] = clean_tokens
            self.docnum += 1
            if not doc['clean_content']: doc['clean_content'] = "."
            # doc = doc if doc['clean_content'] else {}
        except Exception as e:
            LOGGER.exception(e)
            doc = {}
        return doc

    def __call__(self, doc, **kwargs):
        return self.process_item(doc, **kwargs)


class GensimDictionary(object):
    '''Stores a reference to the file containing a gensim dictionary.
    '''
    category = 1

    dictionary_choices = (('1', 'Standard'), )
    STANDARD = '1'

    @staticmethod
    def clean(dictionary):
        """Cleans a dictionary
            (1) Remove stopwords
            (2) Remove hapaxes
            (3) Remove unwanted tokens
            (4) Remove gaps in id sequence after words that were removed
        """
        stoplist = STOPWORDS
        stop_ids = [
            dictionary.token2id[stopword]
            for stopword in stoplist if stopword in dictionary.token2id
        ]
        hapax_ids = [
            tokenid
            for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1
        ]
        unwanted_tokens = stop_ids + hapax_ids
        dictionary.filter_tokens(unwanted_tokens)
        dictionary.compactify()
        return dictionary

    def __init__(self, dataset_id, document_list=None):
        self.dataset_id = dataset_id
        if document_list is not None: self.document_list = document_list

    @mproperty
    def data(self):
        return self.fit_transform()

    @mproperty
    def path(self):
        """Creates a path for accessing this model's data."""
        base_path = conf.PATH_DICTIONARIES
        fname = '_'.join([str(self.dataset_id), 'dictionaries', str(
            self.category)])
        fname = '{}.gsdict'.format(fname)
        return os.path.join(base_path, fname)

    @mproperty
    def document_list(self):
        """
        Get a list of documents from disk or create one.
        """
        document_list_wrapper = ClusterDocumentList(self.dataset_id)
        return document_list_wrapper.fit_transform()

    @document_list.setter
    def document_list(self, value):
        self._document_list = value

    def fit_transform(self):
        """Gets or creates a dictionary."""
        try:
            dictionary = self.get()
        except Exception as e:
            dictionary = self._fit_transform()
        return dictionary

    def get(self):
        """Retrieves an existing dictionary."""
        path = self.path
        dictionary = self.load(path)
        #   -----------------------------------
        msg = 'Found an existing {} with {1} entries'.format(
            colored('dictionary', 'yellow'), colored(
                str(len(dictionary)), 'red'))
        LOGGER.info(msg)
        #   -----------------------------------
        return dictionary

    def delete(self):
        """Deletes an existing dictionary."""
        path = self.path
        if os.path.exists(path): os.remove(path)

    def _fit_transform(self, df=None):
        """Cleans the documents and then constructs a dictionary from them.
            (1) Build a gensim corpus dictionary
            (2) Get a list of documents from disk or create one.
            (3) Iterate through each document to update the dictionary
            (4) Store the dictionary
        """
        #   -----------------------------------
        msg = 'Creating a {}...'.format(colored('Gensim dictionary',
                                                 'yellow'))
        LOGGER.info(msg, extra=dict(border=True))
        #   -----------------------------------
        dictionary = gensim_corpora.Dictionary()
        document_list = self.document_list
        if (document_list is None) or document_list.empty:
            raise NoDataError
        documents = document_list.clean_tokens.values
        for word_list in documents:
            dictionary.doc2bow(word_list, allow_update=True)
        self.clean(dictionary)
        self.save(dictionary)
        return dictionary

    def load(self, path):
        """Loads an existing dictionary."""
        with open(path, mode='rb') as dictionary_file:
            dictionary = pickle.load(dictionary_file)
            return dictionary

    def save(self, dictionary):
        """Pickles a gensim dictionary.
        """
        path = self.path
        with open(path, 'wb') as f:
            pickle.dump(dictionary, f)


class GensimCorpus(object):
    '''Stores a reference to the file containing a gensim corpus.
    '''
    corpus_choices = (('1', 'Standard'), ('2', 'TF-IDF'), ('3', 'LSI-Space'), )
    STANDARD = '1'
    TFIDF = '2'
    LSI_SPACE = '3'

    category = 1

    def __init__(self,
                 dataset_id,
                 num_topics=200,
                 dictionary=None,
                 document_list=None, ):
        self.dataset_id = dataset_id
        #   Length of the corpus
        self.size = 0
        self.num_topics = num_topics
        self.document_dict = {}
        if dictionary is not None: self.dictionary = dictionary
        if document_list is not None: self.document_list = document_list

    @mproperty
    def data(self):
        return self.fit_transform()

    @mproperty
    def path(self):
        """Creates a path for accessing this model's data."""
        base_path = conf.PATH_CORPORA
        fname = '_'.join([str(self.dataset_id), 'corpus', str(
            self.category)])
        fname = '{}.gscorp'.format(fname)
        return os.path.join(base_path, fname)

    @mproperty
    def dictionary(self):
        dictionary_wrapper = GensimDictionary(
            self.dataset_id,
            document_list=self.document_list)
        return dictionary_wrapper.get_or_create()

    @dictionary.setter
    def dictionary(self, value):
        self._dictionary = value

    @mproperty
    def document_list(self):
        document_list_wrapper = ClusterDocumentList(self.dataset_id)
        return document_list_wrapper.fit_transform()

    @document_list.setter
    def document_list(self, value):
        self._document_list = value

    def fit_transform(self):
        """Gets or creates a corpus."""
        try:
            corpus = self.get()
        except Exception as e:
            corpus = self._fit_transform()
        return corpus

    def get(self):
        """Gets a corpus."""
        path = self.path
        corpus = self.load(path)
        #   -----------------------------------
        msg = 'Found an existing {} with {1} entries'.format(
            colored('standard corpus', 'yellow'), colored(
                str(len(corpus)), 'red'))
        LOGGER.info(msg)
        #   -----------------------------------
        return corpus

    def _fit_transform(self):
        """Creates a corpus by iterating one line at a time over items retrieved from
        the database.

        Creates a dictionary if one does not already exist, and then constructs
        a corpus from the documents according to the dictionary entries.
        """
        try:
            #   -----------------------------------
            msg = 'Creating a {}...'.format(colored('standard Gensim corpus',
                                                     'yellow'))
            LOGGER.info(msg, extra=dict(border=True))
            #   -----------------------------------
            dictionary = self.dictionary
            document_list = self.document_list
            if (document_list is None) or document_list.empty:
                raise NoDataError
            documents = document_list.clean_tokens.values
            #   Create corpus from dictionary
            corpus = [dictionary.doc2bow(word_list) for word_list in documents]
            self.save(corpus)
        except Exception as e:
            LOGGER.error(e)
        return corpus

    def to_csv(self, to_csv=True):
        """
        term_list = DataFrame(dict(dictionary))
        """
        #   --------------------------------------------
        msg = 'Getting human format for corpus...'
        LOGGER.info(msg)
        #   --------------------------------------------
        term_list = Series(self.dictionary)
        #   Word X Document matrix
        M = gensim.matutils.corpus2dense(self.data, len(term_list))
        df = DataFrame(M, index=term_list, columns=self.document_list._id)
        if to_csv:
            #   --------------------------------------------
            msg = 'Saving corpus in human format to csv...'
            LOGGER.info(msg)
            #   --------------------------------------------
            base_path = conf.PATH_MATRICES
            fname = '{}_{}_matrix.csv'.format(self.dataset_id, self.category)
            path = os.path.join(base_path, fname)
            df.to_csv(path)
        return df

    def delete(self):
        """Deletes a corpus."""
        path = self.path
        if os.path.exists(path): os.remove(path)

    def load(self, path):
        """Loads an existing corpus in Matrix Market format."""
        corpus = gensim_corpora.MmCorpus(path)
        return corpus

    def save(self, corpus):
        """Pickles a gensim corpus to a standard file path.
        """
        path = self.path
        gensim_corpora.MmCorpus.serialize(path, corpus)

    def get_document_dict(self, corpus=None):
        """Returns a dictionary with integers as keys and corpus entries as
        values.
        """
        if not corpus: corpus = self.fit_transform()
        document_dict = dict(enumerate(corpus))
        self.document_dict = document_dict
        return document_dict


class GensimTFIDFCorpus(GensimCorpus):
    category = 2

    def __init__(self, *args, **kwargs):
        standard_corpus = kwargs.pop('standard_corpus', None)
        if standard_corpus is not None: self.standard_corpus = standard_corpus
        super(GensimTFIDFCorpus, self).__init__(*args, **kwargs)

    @mproperty
    def standard_corpus(self):
        standard_corpus_wrapper = GensimCorpus(
            self.dataset_id,
            num_topics=self.num_topics,
            document_list=self.document_list,
            dictionary=self.dictionary, )
        return standard_corpus_wrapper.fit_transform()

    @standard_corpus.setter
    def standard_corpus(self, value):
        self._standard_corpus = value

    def get(self):
        """Gets a tf-idf corpus."""
        path = self.path
        corpus = self.load(path)
        #   -----------------------------------
        msg = 'Found an existing {} with {1} entries'.format(
            colored('TF.IDF corpus', 'yellow'), colored(
                str(len(corpus)), 'red'))
        LOGGER.info(msg)
        #   -----------------------------------
        return corpus

    def _fit_transform(self):
        """Creates a corpus if one does not already exist, and then transforms
        the entries according to their tf.idf (term-frequency * inverse-document-
        frequency) weights.

        From the gensim docs:
            Compute tf-idf by multiplying a local component (term frequency) with a
            global component (inverse document frequency), and normalizing the resulting
            documents to unit length.
            Formula for unnormalized weight of term i in document j in a corpus of D documents:

                weight_{i,j} = frequency_{i,j} * log_2(D / document_freq_{i})

            or, more generally:

                weight_{i,j} = wlocal(frequency_{i,j}) * wglobal(document_freq_{i}, D)

        """
        #   -----------------------------------
        msg = 'Creating a {}...'.format(colored('TF-IDF corpus', 'yellow'))
        LOGGER.info(msg, extra=dict(border=True))
        #   -----------------------------------
        standard_corpus = self.standard_corpus
        tfidf = gensim_models.TfidfModel(standard_corpus, normalize=True)
        tfidf_corpus = tfidf[standard_corpus]
        self.save(tfidf_corpus)
        return tfidf_corpus

