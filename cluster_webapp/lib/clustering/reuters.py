
import pandas as pd
from pandas import DataFrame
from nltk.corpus import reuters

#	Finance terms
CATS = [
	'dlr', 'dmk', 'earn', 
	'gnp', 'income', 'interest', 
	'money-fx', 'money-supply', 
	'nzdlr', 'yen'
]

def main():
	docs = {}
	for cat in CATS:
		print cat
		fileids = reuters.fileids(cat)[:25]
		print '\t', ', '.join(fileids)
		_docs = dict( ( (fileid, reuters.raw(fileids=fileid)) for fileid in fileids ) )
		docs.update(_docs)
	docs = Series(docs)
	df = DataFrame(docs, columns=['content'])
	return df