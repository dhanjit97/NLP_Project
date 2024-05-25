import customtkinter as ctk
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import os

nltk.download('stopwords')
nltk.download('punkt')

class SpamClassifierApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Spam Classifier")
        self.geometry("500x400")

        self.ps = PorterStemmer()
        self.stopwords = set(stopwords.words('english'))

        self.model = None
        self.vectorizer = None

        self.label = ctk.CTkLabel(self, text="Enter a message:")
        self.label.pack(pady=10)

        self.text_entry = ctk.CTkEntry(self, width=400)
        self.text_entry.pack(pady=10)

        self.classify_button = ctk.CTkButton(self, text="Classify", command=self.classify_text)
        self.classify_button.pack(pady=10)

        self.result_label = ctk.CTkLabel(self, text="")
        self.result_label.pack(pady=10)

    def load_model_and_vectorizer(self, model_path, vectorizer_path):
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
        else:
            print("Model or vectorizer file not found!")

    def transform_text(self, text):
        text = text.lower()
        text = nltk.word_tokenize(text)
        text = [word for word in text if word.isalnum()]
        text = [word for word in text if word not in self.stopwords and word not in string.punctuation]
        text = [self.ps.stem(word) for word in text]
        return " ".join(text)

    def classify_text(self):
        if self.model and self.vectorizer:
            text = self.text_entry.get()
            transformed_text = self.transform_text(text)
            vectorized_text = self.vectorizer.transform([transformed_text])
            prediction = self.model.predict(vectorized_text)
            result = "Spam" if prediction[0] == 1 else "Ham"
            self.result_label.configure(text=f"Result: {result}")
        else:
            self.result_label.configure(text="Model or vectorizer not loaded!")

if __name__ == "__main__":
    app = SpamClassifierApp()
    app.load_model_and_vectorizer("saved_models/KNeighbors.joblib", "saved_models/tfidf_vectorizer.joblib")
    app.mainloop()
