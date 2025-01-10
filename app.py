"""
TensorQuest Hackathon: Problem Statement 1 (SMS SPAM DETECTOR)
Exposing the model built by Team Neural Nexus through Web API and an interface

(c) Aritro 'sortira' Shome
Disclaimer: the owner of this repository does not own nor claims credit for the model used
"""




from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import pickle, nltk, string
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

members = ['Arkapravo Das', 'Subhranil Nandy', 'Harsh Raj Gupta', 'Sayan Roy', 'Aritro Shome']
roles = ['developer', 'developer', 'developer', 'developer', 'developer']
model = load_model('model.h5')
nltk.download('punkt_tab')
nltk.download('stopwords')
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)
    
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)
    
def check_spam(text):
    text = transform_text(text)
    sms_sequence = tokenizer.texts_to_sequences(text)
    sms_padded = pad_sequences(sms_sequence)
    prediction = model.predict(sms_padded)
    prob = prediction[0][0]
    return {"spam_probability": prob, "not_spam_probability": 1.0 - prob};
    
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/team')
def team():
    return render_template('team.html', roles=roles, members=members)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        sms_text = request.form['sms_text']
        result = check_spam(sms_text)
        return render_template('result.html', sms_text=sms_text, result=result)
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
