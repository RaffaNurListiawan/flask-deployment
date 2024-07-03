from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd 
from joblib import dump, load

app = Flask(__name__)

model = pickle.load(open('trained_model.pkl', 'rb'))

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
        'self_esteem': [request.form['self_esteem']],
        'mental_health_history': [request.form['mental_health_history']],
        'depression': [request.form['depression']],
        'headache': [request.form['headache']],
        'blood_pressure': [request.form['blood_pressure']],
        'sleep_quality': [request.form['sleep_quality']],
        'breathing_problem': [request.form['breathing_problem']],
        'noise_level': [request.form['noise_level']],
        'living_conditions': [request.form['living_conditions']],
        'safety': [request.form['safety']],
        'basic_needs': [request.form['basic_needs']],
        'academic_performance': [request.form['academic_performance']],
        'study_load': [request.form['study_load']],
        'teacher_student_relationship': [request.form['teacher_student_relationship']],
        'future_career_concerns': [request.form['future_career_concerns']],
        'social_support': [request.form['social_support']],
        'peer_pressure': [request.form['peer_pressure']],
        'extracurricular_activities': [request.form['extracurricular_activities']],
        'bullying': [request.form['bullying']],
    }
        data_df = pd.DataFrame(new_data)
        predict = model.predict(data_df)

        print(predict)

        return render_template('result.html', result=predict[0])
