

from flask import Flask, render_template, request
import pickle
import numpy as np

import os
os.getcwd
os.chdir('C:\\Users\\ABHI ALEXY\\Desktop\\vs code\\IRIS_Flower_Detection')


model = pickle.load(open('iri.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['GET','POST','DELETE'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    arr = np.array([[data1, data2, data3, data4]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)


if __name__ == "__main__":
   app.run()
   # app.run(debug=True)


