import random
import sqlite3
import nltk
import pandas as pd
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

responses = {
    "hello": "Hello there!",
    "bye": "Goodbye!",
    "code": "What type of code would you like me to generate?"
}

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.form["message"].lower()
    language = request.form["language"]
    response = None

    if user_message in responses:
        response = responses[user_message]
    else:
        response = generate_response(user_message)
    
    if "code" in user_message:
        code = generate_code(language, user_message)
        return jsonify({"response": response, "code": code})
    return jsonify({"response": response})

def generate_response(user_message):
    words = word_tokenize(user_message)
    
    for word in words:
        if word in responses:
            return responses[word]
    return random.choice(responses.values())

def generate_code(language, user_message):
    conn = sqlite3.connect("chatbot.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS codes (language TEXT, code TEXT);")
    cursor.execute("SELECT * FROM codes WHERE language = ?", (language,))
    data = cursor.fetchall()
    data = pd.DataFrame(data, columns=["language", "code"])
    if data["code"].empty:
        return "No code found for this language."
    else:
        tfidf = TfidfVectorizer().fit_transform(data["code"])
        knn = NearestNeighbors(n_neighbors=1).fit(tfidf)
        user_message_vector = TfidfVectorizer().fit_transform([user_message])
        distances, indices = knn.kneighbors(user_message_vector)

if __name__ == "__main__":
    app.run(debug = True)