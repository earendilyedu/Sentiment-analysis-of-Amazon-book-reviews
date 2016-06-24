import nltk
from sentence import Sentence

class Review(object):
	def __init__(self, review_text):
		self.review_text = review_text
		#convert unicode to string
		self.review_text = self.review_text.encode('ascii','ignore')
		# splitter for converting a review to a list of sentences.
		# each sentence is a string or uni and sparated by comma like: [u'Spiritually and ....', u'A book that .....']
		nltk_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
		self.sentences  = nltk_splitter.tokenize(self.review_text)
      

	def sentence_tokenize(self):
		"""
		INPUT: a list of sentences after nltk_splitter
		OUTPUT: List of Sentence objects

		apply sentence class for each sentence in a reivew
		"""
		return [Sentence(sent) for sent in self.sentences]
