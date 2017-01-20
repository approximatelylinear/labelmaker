
#   Stdlib
import copy
import math
from functools import partial

#   3rd party
from numpy import log2, log1p
from sklearn.preprocessing import normalize as sk_normalize
from gensim.corpora import dictionary as gensim_dictionary
from gensim.models.tfidfmodel import TfidfModel as gensim_tfidfmodel
from gensim.matutils import corpus2csc, corpus2denseX


class GensimPipeline(object):
    def __init__(self, submodels, attrs):
        self.submodels = submodels
        sefl.attrs = attrs

    def get_subattr(self, name):
        """
        a__b__c
        """
        submodels = self.submodels
        parts = name.split(u'__')
        parents = parts[:-1]
        attr = parts[-1]
        curr_submodel = None
        while parents:
            parent = parents.pop()
            for submodel in submodels:
                if submodel.name == parent:
                    curr_submodel = submodel
                    submodels = submodel.submodels
        val = getattr(curr_submodel, attr, None) if curr_submodel else None
        return val

    def fit(self, X):
        submodels = self.submodels
        for submodel in submodels[:-1]:
            params = { self.get_subattr(attr) for attr in (self.attrs.get(submodel.name) or []) }
            submodel.fit_transform(X, **params)
        submodel = submodels[-1]
        params = { self.get_subattr(attr) for attr in self.attrs.get(submodel.name) }
        submodel.fit(X, **params)


def test():
    class A(object):
        def __init__(self):
            self.name = 'A'
            self.a = None

        def fit_transform(self, X, **kwargs):
            self.fit(X, **kwargs)
            X_trans = self.transform(X, **kwargs)
            return X_trans

        def fit(self, X, **kwargs):
            print("Fitting")
            self.a = 'a'

        def transform(self, X, **kwargs):
            return X

    class B(object):
        def __init__(self):
            self.name = 'B'
            self.b = None
            self.submodels = [ A() ]

        def fit_transform(self, X, **kwargs):
            self.fit(X, **kwargs)
            X_trans = self.transform(X, **kwargs)
            return X_trans

        def fit(self, X, **kwargs):
            print("Fitting")
            self.b = 'b'

        def transform(self, X, **kwargs):
            return X

    class C(object):
        def __init__(self):
            self.name = 'C'
            self.c = None
            self.submodels = []

        def fit_transform(self, X, **kwargs):
            self.fit(X, **kwargs)
            X_trans = self.transform(X, **kwargs)
            return X_trans

        def fit(self, X, **kwargs):
            print("Fitting")
            self.c = 'c'

        def transform(self, X, **kwargs):
            return X

    pipeline = GensimPipeline(
        submodels=[ B(), C() ],
        attrs={
            'B': 'A__a',
            'C': 'B__b',
        }
    )


"""
model:
    -
        definition: *x
        parameters:
    -
        definition: *y
        parameters:
            - x__a
            - x__b
            - x__submodel1__a
    -
        definition: *z
        parameters:
"""


class Dictionary(object):
    def __init__(self,
        max_df=.5,
        min_df=1,
        max_features=None,
        binary=False,
        stoplist=None,
    ):
    """
    #   Initial data
    from sklearn.datasets import fetch_20newsgroups
    newsgroups_train = fetch_20newsgroups(subset='train')
    data = newsgroups_train.data[:20]
    tokens = [d.split(' ') for d in data]

    #   Stoplist
    stoplist = ['lower-risk', 'Why', 'may']

    #   Gensim alone
    gdict = gensim_dictionary.Dictionary()
    gdict.add_documents(tokens)
    corpus = ( gdict.doc2bow(doc, allow_update=False, return_missing=False) for doc in tokens )
    X_trans = corpus2csc(corpus)

    #   Wrapper class
    #       Basic functionality
    mydict = Dictionary()
    mydict.fit(tokens)
    assert mydict.model.token2id
    assert mydict.transform([['a', 'the', 'sign']])
    for w in stoplist:
        #   No stoplist passed into constructor, so these should be present
        assert (w in mydict.model.token2id)

    #       Basic functionality + stoplist
    mydict = Dictionary(stoplist=stoplist)
    mydict.fit(tokens)
    for w in stoplist:
        #   Stoplist passed into constructor, so these should be absent
        print(w in mydict.model.token2id)

    """
        self._max_df = max_df
        self._min_df = min_df
        self._max_features = max_features
        self._stoplist = stoplist
        self.model_callable = gensim_dictionary.Dictionary
        #   Development
        self.data = {}
        self.name = self.__class__.__name__
        self.config = {}
        self.model = self._init_model()

    def _init_model(self, globals_=None):
        """
        Create this model by instantiating the model callable with the parameters.
        """
        if globals_ is None:
            globals_ = globals()
        config = self.config
        model = self.model_callable()
        return model

    def inverse_transform(self, X):
        #   TBD: Use `self.model.id2token`
        pass

    def fit(self, raw_documents, y=None):
        #   Clean the model
        self._update_fit(raw_documents, y=y)

    def _update_fit(self, raw_documents, y=None):
        self.model.add_documents(raw_documents)
        if self._stoplist is not None:
            self.filter_stopwords()
        if isinstance(self._max_df, int):
            #   Must be a fraction of the total
            max_df = self._max_df / (self.model.num_docs + 0.0)
        if isinstance(self._min_df, float):
            #   Must be an absolute number
            min_df = int(self._min_df * self.model.num_docs)
        self.model.filter_extremes(
            no_below=self._min_df,
            no_above=self._max_df,
            keep_n=self._max_features
        )
        self.model.compactify()

    def transform(self, raw_documents, copy=True):
        doc2bow = self.model.doc2bow
        corpus = ( doc2bow(doc, allow_update=False, return_missing=False) for doc in raw_documents )
        X_trans = corpus2csc(corpus)
        self.data['X_trans'] = X_trans
        return X_trans

    def fit_transform(self, raw_documents, y=None):
        self.fit(raw_documents, y=y)
        return self.transform(raw_documents, y=y)

    def predict(self, raw_documents, y=None):
        raise NotImplementedError

    def fit_predict(self, raw_documents, y=None):
        raise NotImplementedError

    def get_stop_words(self):
        return self._stoplist

    def filter_stopwords(self):
        stoplist = self._stoplist
        token2id = self.model.token2id
        stop_ids = [ token2id[stopword] for stopword in stoplist if stopword in token2id ]
        self.model.filter_tokens(stop_ids)

    def get_feature_names(self):
        return [ name for name, _ in sorted(self.token2id.iteritems(), key=lambda x: x[1]) ]



class Tfidf(object):

    def __init__(self):
        self.model_callable = gensim_tfidfmodel
        #   Development
        self.data = {}
        self.name = self.__class__.__name__
        self.config = {}
        self.model = self._init_model()

    def _init_model(self, globals_=None):
        """
        Create this model by instantiating the model callable with the parameters.
        """
        if globals_ is None:
            globals_ = globals()
        config = self.config
        model = None
        return model

    def _init_parameters(self, **params):
        # parameters = super(Tfidf, self)._init_parameters()
        #################
        #   For development
        parameters = {
            self.name: params
        }
        #################
        parameters_curr = parameters[self.name]
        #   Translate parameters to Gensim format
        sublinear_tf = parameters_curr.pop('sublinear_tf', None)
        if sublinear_tf:
            #   1+x (base e)
            parameters_curr['wlocal'] = log1p
        # Default wglobal: log_2(total_docs / doc_freq)
        use_idf = parameters_curr.pop('use_idf', True)
        if use_idf:
            smooth_idf = parameters_curr.pop('smooth_idf', False)
            #   Normalize by inverse document frequency
            if smooth_idf:
                #   Smooth by adding 1 to each document frequency
                parameters_curr['wglobal'] = lambda x, y: log2(x, y + 1)
        else:
            #   Set global normalization to the identity
            parameters_curr['wglobal'] = lambda x, y: 1
        norm = parameters_curr.pop('norm')
        if norm == 'l1'
            parameters_curr['normalize'] = partial(sk_normalize, norm='l1')
        elif norm == 'l2':
            parameters_curr['normalize'] = partial(sk_normalize, norm='l2')
        elif callable(norm):
            parameters_curr['normalize'] = norm
        return parameters

    def fit(self, corpus=None, id2word=None, dictionary=None, y=None):
        params = copy.deepcopy(self.parameters[self.name])
        params['corpus'] = corpus
        params['id2word'] = id2word
        params['dictionary'] = dictionary

        self.model = self.model_callable(**params)

    def transform(self, X, y=None, as_matrix=False):
        X_trans = self.model[X]
        if as_matrix:
            X_trans = corpus2csc(X_trans)
            self.data['X_trans'] = X_trans
        return X_trans

    def fit_transform(self, X, id2word=None, dictionary=None, y=None, as_matrix=False):
        self.fit(corpus=X, id2word=id2word, dictionary=dictionary, y=y)
        X_trans = self.transform(X)
        return X_trans



class Lsi(object):
    def _init_parameters(self):
        parameters = super(Lsi, self)._init_parameters()
        parameters_curr = parameters[self.name]

    def fit(self, corpus=None, id2word=None, dictionary=None, y=None):
        params = copy.deepcopy(self.parameters[self.name])
        params['corpus'] = corpus
        params['id2word'] = id2word
        self.model = self.model_callable(**params)


