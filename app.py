from flask import Flask, render_template, redirect, url_for, request,jsonify
from werkzeug.wrappers import Request, Response
import pandas as pd

from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import LatentDirichletAllocation, NMF
from scipy import sparse
from sklearn.metrics import jaccard_score, recall_score, precision_score, accuracy_score, f1_score
import nltk
import re
import string
from nltk.stem import WordNetLemmatizer
import numpy as np
import warnings

import joblib
model_filename = 'models/model.joblib'
vect_filename = 'models/vec_to_tag.joblib'
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# return model predictions
@app.route("/api/predict", methods=["GET"])

def predict():
    msg_data={}
    for k in request.args.keys():
        val=request.args.get(k)
        msg_data[k]=val
    text = msg_data['txt_question'] + msg_data['txt_body']
    
    # import models
    # TODO: telecharger le modèle
    model = joblib.load(model_filename)
    vect = joblib.load(vect_filename)
    # preparation input
    formated_input = pd.Series([word_processing(text)])
    # prediction
    arr_results = model.predict(formated_input).todense()
    
    arr_results = np.insert(arr_results, 0, 1)
    arr_results = np.insert(arr_results, 222, 1)
    
    # conversion vecteur en tags
    arr_results = vect.inverse_transform(arr_results)[0].tolist()
    if len(arr_results)==0:
        msg_result = 'No tag found...'
    else:
        msg_result = ' | '.join(arr_results)
    return msg_result
    
def get_stop_words():
    stop_words = set()
    useless_words = ['th','utc','am','pm']
    default = nltk.corpus.stopwords.words('english')
    stop_words.update(default)
    stop_words.update(useless_words)
    return stop_words

def word_processing(x):
    remove_unicode = lambda x: x.encode("ascii", "ignore").decode()
    text = remove_unicode(x)

    lower_text = lambda x: x.lower()
    text = lower_text(text)

    no_number_text = lambda x: ''.join([i for i in x if not i.isdigit()])
    text = no_number_text(text)

    no_punct_text = lambda x: x.translate(str.maketrans('', '', string.punctuation))
    text = no_punct_text(text)

    words = nltk.tokenize.word_tokenize(text)
    stop_words = get_stop_words()
    words_filtered = []
    for w in words:
        if w not in stop_words:
            words_filtered.append(w)
    
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = []
    text = nltk.pos_tag(words_filtered)
    for w in text:
        if w[1] in ['NN','NNP','NNPS']:
            lemmatized_words.append(lemmatizer.lemmatize(w[0]))
    formated_text = ', '.join(lemmatized_words)
    return formated_text

def tokenize_text(x):
    x=x.split(', ')
    tags=[i.strip() for i in x]
    return tags

if __name__ == "__main__":
    app.run()