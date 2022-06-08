from senti_lib import PredictSentiment


class DashBoard:
	def predict(self):
		pred = PredictSentiment()
		review_text = input("Enter The Review")

		output = pred.input(review_text)

		print (review_text, " :",output)

		return None

if __name__ == "__main__":
	db = DashBoard()
	db.predict()


