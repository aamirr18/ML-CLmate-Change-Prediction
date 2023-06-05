#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__) #Initialize the flask App
model = pickle.load(open('climate_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('climate_index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0],2)

    return render_template('climate_index_2.html', prediction_text='Yield for the given values is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)

