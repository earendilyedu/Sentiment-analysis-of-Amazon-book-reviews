import pickle

class SentimentModel(object):

	SENTIMENT_MODEL = pickle.load(open('/Users/Louis/final-project/classes/main/models/senti_model.pkl', 'rb'))

	def get_positive_proba(self, sent):
		return SentimentModel.SENTIMENT_MODEL.predict_proba(sent.get_features(asarray=True))[0][1]

class OpinionModel(object):

	OPINION_MODEL = pickle.load(open('/Users/Louis/final-project/classes/main/models/opin_model.pkl', 'rb'))

	def get_opinionated_proba(self, sent):
		return OpinionModel.OPINION_MODEL.predict_proba(sent.get_features(asarray=True))[0][1]
