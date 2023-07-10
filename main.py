from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import pickle
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import random
import json
import speech_recognition as sr
import pyttsx3

stemmer = LancasterStemmer()
model = tf.keras.models.load_model('saved_model/my_model')

with open("data.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)

with open("intents.json", encoding="utf8") as file:
    data = json.load(file)

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)

def speak(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    female_voices = [v for v in voices if 'female' in v.name.lower()]
    if len(female_voices) > 0:
        engine.setProperty('voice', female_voices[0].id)
    else:
        print("No female voices found. Using default voice.")
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    engine.say(text)
    engine.runAndWait()

def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1.5
        r.energy_threshold = 300
        audio = r.listen(source, timeout=5)

    try:
        print("Recognizing...")
        query = r.recognize_google(audio)
        print(f"User said: {query}\n")
        return query
    except Exception as e:
        print("Unable to recognize your voice.")
        return ""

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/send-message", methods=['POST'])
def api_send_message():
    message = request.json["message"]

    if message in ['bye', 'break', 'exit', 'quit', 'close']:
        for tg in data["intents"]:
            if tg["tag"] == message:
                responses = tg["responses"]
        response = random.choice(responses)
    else:
        results = np.array([bag_of_words(message, words)])
        results = model.predict(results)[0]
        result_index = np.argmax(results)
        tag = labels[result_index]

        if results[result_index] > 0.70:
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responses = tg["responses"]
            response = random.choice(responses)
        else:
            response = "Sorry, I didn't understand that. I'm limited to a specific purpose!"

    return jsonify({"response": response})

@app.route("/api/send-voice-message", methods=['POST'])
def api_send_voice_message():
    message = listen()

    if message in ['bye', 'break', 'exit', 'quit', 'close']:
        for tg in data["intents"]:
            if tg["tag"] == message:
                responses = tg["responses"]
        response = random.choice(responses)
    else:
        results = np.array([bag_of_words(message, words)])
        results = model.predict(results)[0]
        result_index = np.argmax(results)
        tag = labels[result_index]

        if results[result_index] > 0.70:
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responses = tg["responses"]
            response = random.choice(responses)
        else:
            response = "Sorry, I didn't understand that. I'm limited to a specific purpose!"

    speak(response)
    return jsonify({"response": response})

@app.route("/api/speak", methods=['POST'])
def api_speak():
    text = request.json["text"]
    speak(text)
    return jsonify({"success": True})

if __name__ == "__main__":
    app.run(port=5000, debug=False)
