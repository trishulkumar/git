#!/usr/bin/env python
# coding: utf-8




# In[12]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


# In[13]:


app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))


# In[14]:


@app.route('/')
def home():
    return render_template('index.html')


# In[15]:


@app.route('/predict',methods=['POST'])

def predict():
    # for rendering results on HTML GUI
    data1=request.form["Danceability"]
    data2=request.form["Energy"]
    data3=request.form["Loudness"]
    data4=request.form["Speechiness"]
    data5=request.form["Acousticness"]
    data6=request.form["Instrumentalness"]
    data7=request.form["Tempo"]
    data8=request.form["Valence"]
    data9=request.form["Liveness"]
    data10=request.form["Genre"]
    
    arr=[data1,data2,data3,data4,data5,data6,data7,data8,data9,data10]
    output=model.predict([arr])
    
    return render_template('index.html',prediction_text='The probability of song being on billboard Top100 is {}'.format(output))


# In[16]:


if __name__== "__main__":
    app.run(debug=True)

