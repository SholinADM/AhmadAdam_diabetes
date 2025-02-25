#%%Hehe
import streamlit as st
import pandas as pd
import pickle
import os
from PIL import Image
import numpy as np
#MODEL PATH
MODEL_PATH =os.path.join(os.getcwd(),'model','model.pkl')
with open(MODEL_PATH,'rb') as file:
    model = pickle.load(file)

#IMAGE PATH
IMAGE_PATH = os.path.join(os.getcwd(),'static','sugar.jpg')
image = Image.open(IMAGE_PATH)
#%%
st.title('Diabetes Predictor 3000')
st.header('we are aiming to help new user to predict if they have diabetes or not')
st.subheader('What to know about diabetes')
st.write("Diabetes mellitus refers to a group of diseases that affect how the body uses blood sugar (glucose). Glucose is an important source of energy for the cells that make up the muscles and tissues. It's also the brain's main source of fuel.The main cause of diabetes varies by type. But no matter what type of diabetes you have, it can lead to excess sugar in the blood. Too much sugar in the blood can lead to serious health problems. Chronic diabetes conditions include type 1 diabetes and type 2 diabetes. Potentially reversible diabetes conditions include prediabetes and gestational diabetes. Prediabetes happens when blood sugar levels are higher than normal. But the blood sugar levels aren't high enough to be called diabetes. And prediabetes can lead to diabetes unless steps are taken to prevent it. Gestational diabetes happens during pregnancy. But it may go away after the baby is born...source: Mayo Clinic Website")

st.image(image, use_container_width=True)

st.sidebar.header("Check if you're diabetic")
with st.sidebar:
    with st.form(key='my_form'):
        Glucose = st.number_input("Glucose",0,300,120)
        BloodPressure = st.number_input("Diastolic Blood Pressure (mm Hg)",0,200,75)
        SkinThickness = st.number_input("Skin Thickness (mm)",0,100,30)
        Insulin = st.number_input("Insulin Level (mu U/ml)",0,800,160)
        BMI = st.number_input("BMI (kg/m^2)",0,100,25)
        Age = st.number_input("Age (Years)",0,200,32)
        DiabetesPedigreeFunction = st.slider("Diabetes Pedigree Function",0.0,3.0,0.0001)
        submit = st.form_submit_button("Submit")

#Prediction Section
st.subheader("Will you have Diabetes?")
if submit:
    new_data = np.expand_dims([Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age],axis = 0)
    outcome = model.predict(new_data)[0]
    if outcome == 0:
        st.success("Congratulation!!! You are healthy! 🤩🤩🤩")
        st.write("Keep your healthy lifestyle.")
        st.balloons()
    else:
        st.warning(" ❗️❗️❗️YOU HAVE A HIGH RISK OF GETTING DIABETES!!! 😱😱😱")
        st.write("Run away!!! You can't escape once you sick but you can get healthier now.")
        st.snow()

