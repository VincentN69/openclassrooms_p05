from flask import Flask, render_template, redirect, url_for, request,jsonify
from werkzeug.wrappers import Request, Response
import pandas as pd

import joblib
model_filename = 'models/model.joblib'

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
    input_df = pd.DataFrame(text, columns=['Text'], index=[0])
    
    # TODO: telecharger le modèle
    #model = joblib.load(open(model_filename, "rb"))
    #arr_results = model.predict(input_df)
    arr_results = ["rien","rien","rien"]
    msg_result = ''
    msg_result = ' | '.join(arr_results)
    return msg_result
    

if __name__ == "__main__":
    app.run()