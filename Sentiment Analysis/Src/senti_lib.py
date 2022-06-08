import cherrypy
import pickle
import pandas as pd
import nltk
from nltk.tokenize import TweetTokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report

class PredictSentiment:
	def __init__(self):
		self.model = None
		self.vectorizer = None
		with open("lstm_model_old", 'rb') as data:
			self.model = pickle.load(data)
		with open("vectorizer_old", 'rb') as data:
			self.vectorizer = pickle.load(data)

	def lemma(self,text):
		w_tokenizer = TweetTokenizer()
		lemmatizer = nltk.stem.WordNetLemmatizer()
		return " ".join([lemmatizer.lemmatize(w) for w  in w_tokenizer.tokenize(text)])

	def input(self,review_text):

		sample_dict = {"Processed_Phrase":[review_text]}
		sample_df = pd.DataFrame(sample_dict)

		sample_df["Processed_Phrase"] = sample_df["Processed_Phrase"].str.lower()
		sample_df["Processed_Phrase"] = sample_df.Processed_Phrase.apply(self.lemma)


		x_test = self.vectorizer.texts_to_sequences(sample_df["Processed_Phrase"])
		x_test = pad_sequences(x_test, maxlen=50)


		
		y_test_pred=self.model.predict_classes(x_test)
		final_pred = y_test_pred[0]

		if final_pred == 0:
			return "Negative"
		elif final_pred == 1:
			return "Somewhat Negative"
		elif final_pred == 2:
			return "Neutral"
		elif final_pred == 3:
			return "Somewhat Positive"
		else:
			return "Positive"


if __name__ == "__main__" :
	pred = PredictSentiment()
	print (pred.input("This Movie is awesome"))


