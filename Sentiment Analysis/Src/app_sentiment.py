import cherrypy
import pickle
import pandas as pd
import nltk
from nltk.tokenize import TweetTokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report


class SentiApp(object):
	"""docstring for SentiApp"""
	def __init__(self):
		self.model = None
		self.vectorizer = None
		with open("lstm_model", 'rb') as data:
			self.model = pickle.load(data)
		with open("vectorizer", 'rb') as data:
			self.vectorizer = pickle.load(data)

	@cherrypy.expose
	def index(self):
		return "Welcome to the Sentiment Analysis Page. Still Improving..."
	@cherrypy.expose
	def generate(self, length=None):
		length = str(length)
		df = pd.DataFrame({"Review_Text_Clean":[length]})

		# tfidf_tr = None
		# model = None

		# print (length)
		feature_test = self.tfidf_tr.transform(df).toarray()
		predictions = self.model.predict(feature_test)
		return predictions

	def lemma(self,text):
		w_tokenizer = TweetTokenizer()
		lemmatizer = nltk.stem.WordNetLemmatizer()
		return " ".join([lemmatizer.lemmatize(w) for w  in w_tokenizer.tokenize(text)])

	def get_output(self,input_df):
		x_test = self.vectorizer.texts_to_sequences(input_df["Processed_Phrase"])
		x_test = pad_sequences(x_test, maxlen=50)


		
		y_test_pred=self.model.predict_classes(x_test)
		
		return y_test_pred


	@cherrypy.expose	
	def review_text_input(self,review_text=None):
		if review_text == None:
			return "Please Enter a Valid Review Sentence"
		else:
			sample_dict = {"Processed_Phrase":[review_text]}
			sample_df = pd.DataFrame(sample_dict)

			sample_df["Processed_Phrase"] = sample_df["Processed_Phrase"].str.lower()
			sample_df["Processed_Phrase"] = sample_df.Processed_Phrase.apply(self.lemma)


			pred_value = self.get_output(sample_df)


			# x_test = self.vectorizer.texts_to_sequences(sample_df["Processed_Phrase"])
			# x_test = pad_sequences(x_test, maxlen=50)


			
			# y_test_pred=self.model.predict_classes(x_test)
			# print (y_test_pred)

			return pred_value

if __name__ == '__main__':
	cherrypy.quickstart(SentiApp())
	# snt = SentiApp()
	# snt.review_text_input("Good Movie")


































# sample_dict = {"Processed_Phrase":["Super Duper Awesome Movie"]}
# sample_df = pd.DataFrame(sample_dict)

# sample_df["Processed_Phrase"] = sample_df["Processed_Phrase"].str.lower()


# w_tokenizer = TweetTokenizer()
# lemmatizer = nltk.stem.WordNetLemmatizer()

# def lemma(text):
#     return " ".join([lemmatizer.lemmatize(w) for w  in w_tokenizer.tokenize(text)])

# sample_df["Processed_Phrase"] = sample_df.Processed_Phrase.apply(lemma)

# vectorizer = None
# model = None
# with open("lstm_model", 'rb') as data:
# 	model = pickle.load(data)
# with open("vectorizer", 'rb') as data:
# 	vectorizer = pickle.load(data)


# x_test = vectorizer.texts_to_sequences(sample_df["Processed_Phrase"])
# x_test = pad_sequences(x_test, maxlen=50)


# from sklearn.metrics import classification_report
# y_test_pred=model.predict_classes(x_test, verbose=1)

# print (y_test_pred)

# # y_test_num = np.argmax(y_test_pred, axis=-1)

# # dum_df = pd.DataFrame({"Y_Test":y_test_num})
# # print (dum_df["Y_Test"].value_counts())