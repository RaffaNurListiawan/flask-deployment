from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd 
from joblib import dump, load

app = Flask(__name__)

model = pickle.load(open('trained_rf.pickle', 'rb'))

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/klasifikasi",  methods=['POST', 'GET'])
def klasifikasi():
    if request.method == 'GET':
        return render_template('klasifikasi.html')
    else:
        new_data = {
        'anxiety_level': [request.form['anxiety_level']],
        'depression': [request.form['depression']],
        'future_career_concerns': [request.form['future_career_concerns']],
        'headache': [request.form['headache']],
        'mental_health_history': [request.form['mental_health_history']],
        'bullying': [request.form['bullying']],
    }
        data_df = pd.DataFrame(new_data)
        predict = model.predict(data_df)

        print(predict)

        return render_template('result.html', result=predict[0])
