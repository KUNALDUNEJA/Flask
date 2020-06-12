# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 01:14:56 2020

@author: KUNAL DUNEJA
"""

from flask import Flask,request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger


app=Flask(__name__)
Swagger(app)
with open('classifier.pkl','rb') as file:
    model=pickle.load(file)

@app.route('/')
def welcome():
    return "WELCOME ALL"


@app.route('/predict')
def predict_note_authentication():
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtoisis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    variance=request.args.get('variance')
    skewness=request.args.get('skewness')
    curtoisis=request.args.get('curtoisis')
    entropy=request.args.get('entropy')
    prediction=model.predict([[variance,skewness,curtoisis, entropy]])
    return "The predicted value is" + str(prediction)
     
     
@app.route('/predict_file',methods=["POST"])
def predict_note_authentication_file():
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """

    df_test=pd.read_csv(request.files.get("file"))
    prediction=model.predict(df_test)
    return "The predicted value is" + str(list(prediction))
     
     






if __name__=='__main__':
    app.run()
