import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error,r2_score
import flask
from flask import request,render_template
from flask import Flask
import pickle

app=Flask(__name__)
with open ("MLR_model.pkl","rb") as f:
    m=pickle.load(f)

@app.route('/')
def main_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def prediction():
    a=[float(i) for i in request.form.values()]
    sol=m.predict([a])[0]
    return render_template("index.html",result=float(sol))
if __name__=='__main__':
    app.run(debug=True)