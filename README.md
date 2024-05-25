Spam Classifier

This project is a spam classifier that uses a machine learning model to classify messages as either spam or ham (not spam).

Installation

1. Clone the repository:

   git clone https://github.com/<your-username>/spam-classifier.git

3. Install the required Python packages:

   pip install -r requirements.txt

5. Download NLTK data:

    python -m nltk.downloader stopwords punkt

Usage

1. Load the model and vectorizer:

   app.load_model_and_vectorizer("saved_models/KNeighbors.joblib", "saved_models/tfidf_vectorizer.joblib")

3. Run the application:

   python spam_classifier_app.py

5. Enter a message in the text box and click the "Classify" button to classify the message as spam or ham.


Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
