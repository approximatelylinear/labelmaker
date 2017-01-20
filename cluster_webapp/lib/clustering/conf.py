
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_DATA = os.path.join(THIS_DIR, 'data')
PATH_CLUSTER_DOCUMENTS = os.path.join(PATH_DATA, 'documents')
PATH_CLUSTER_DOCUMENT_LISTS = os.path.join(PATH_DATA, 'cluster_document_lists')
PATH_DICTIONARIES = os.path.join(PATH_DATA, 'dictionaries')
PATH_CORPORA = os.path.join(PATH_DATA, 'corpora')
PATH_MATRICES = os.path.join(PATH_DATA, 'matrices')
PATH_LSI_SPACES = os.path.join(PATH_DATA, 'lsi_spaces')
PATH_SIMILARITY_INDICES = os.path.join(PATH_DATA, 'similarity_indices')
PATH_RAW_SIMILARITY_INDICES = os.path.join(PATH_DATA, 'raw_similarity_indices')
