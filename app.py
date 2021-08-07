#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from colorama import Fore, Back, Style
  

app = Flask(__name__)
model = pickle.load(open('project.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = np.array(int_features).reshape(1,-1)
    prediction = model.predict(final_features)
    
    if prediction == [0]:
        predict_text = 'Sorry to say,but person will not survive'
    else:
        predict_text = 'Person will survive'

    return render_template('index.html' , prediction_text = predict_text)

if __name__ == "__main__":
    app.run(debug=True)

