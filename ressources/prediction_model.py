import joblib
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

import string
from bs4 import BeautifulSoup

from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy import sparse
import re
import numpy as np

class PredictionModel:
    MODEL_PATH = 'D:/data/P05/models/1_LogisticRegression_TFidf.pkl'
    OUTPUT_TRANSFORMER_PATH = 'D:/data/P05/models/tags_transformer.joblib'
    
    def __init__(self) -> None:
        self._model = self.import_predict_model()
        self._output_model = self.import_ouput_model()
        pass
    def format_input(self, input_dict) -> pd.Series:
        input_text = input_dict['txt_question'] +' '+ input_dict['txt_body']
        input_text = [self.word_processing(input_text)]
        return input_text
    
    def predict(self, X):
        arr_results = []
        # preparation input
        formated_input = self.format_input(X)
        print('informative words :', formated_input)
        
        # prediction
        arr_results = self._model.predict(formated_input)#.todense()
        
        # conversion vecteur en tags
        arr_results = self._output_model.inverse_transform(arr_results)[0].tolist()
        print('tags predicted :',arr_results)
        return arr_results
    
    def import_predict_model(self):
        model = joblib.load(self.MODEL_PATH)
        return model
    
    def import_ouput_model(self):
        model = joblib.load(self.OUTPUT_TRANSFORMER_PATH)
        return model

    def word_processing(self, text):
        
        remove_code = lambda x: re.sub('<code>(.*?)</code>', ' ', x, flags=re.MULTILINE|re.DOTALL)
        text = remove_code(text)

        remove_tags = lambda x : BeautifulSoup(x, 'html.parser').get_text()
        text = remove_tags(text)
        
        remove_unicode = lambda x: x.encode("ascii", "ignore").decode()
        text = remove_unicode(text)

        lower_text = lambda x: x.lower()
        text = lower_text(text)

        no_number_text = lambda x: ''.join([i for i in x if not i.isdigit()])
        text = no_number_text(text)
        
        no_punct_text = lambda x: re.sub(r'(?:[^\w\s]|_)+', ' ', x, flags=re.MULTILINE|re.DOTALL)
        text = no_punct_text(text)

        words = nltk.tokenize.word_tokenize(text)
        stop_words = self.get_stop_words()
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

    def get_stop_words(self):
        stop_words = set()
        useless_words = ['th','utc','am','pm']
        default = nltk.corpus.stopwords.words('english')
        stop_words.update(default)
        stop_words.update(useless_words)
        return stop_words

if __name__ =='__main__':
    test = PredictionModel()
    input_test = {}
    input_test['txt_question'] = 'How to pass two strings to a PowerShell function?'
    input_test['txt_body'] = '<p>How do I pass <code>$dir</code> and <code>$file</code>, then <a href=""https://stackoverflow.com/q/15113413/4531180"">concatenate</a> them to a single path?</p><p>Output:</p><p>worker:</p><pre><code>. /home/nicholas/powershell/functions/library.ps1$dir = &quot;/home/nicholas/powershell/regex&quot;$file = &quot;a.log&quot;SearchFile($dir,$file)</code></pre><p>I seem to be passing an array of sorts for <code>$dir</code> and nothing is getting assigned for <code>$file</code> in <code>SearchFile</code> as expected.</p>'
    test.predict(input_test)