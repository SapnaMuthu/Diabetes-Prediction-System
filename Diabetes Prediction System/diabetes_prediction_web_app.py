# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 16:19:04 2022

@author: 17347
"""

import numpy as np 
import streamlit as st
import pickle 

loaded_model = pickle.load(open("C:/Users/17347/Sapna's Projects/Diabetes Prediction System/diabetes_trained_model.sav",'rb'))

#creating a function for prediction
def diabetes_prediction(input_data):
        input_data_processed = np.asarray(input_data)
        
        #reshape the array as we are predicting only for one instance out of all the 768 samples
        input_data_reshaped = input_data_processed.reshape(1,-1)

        prediction = loaded_model.predict(input_data_reshaped)
        if prediction[0] == 0:
            print("Patient is Non-Diabetic")
        else:
            print("Patient is Diabetic")
            
def main():
    # giving a title 
    st.title('Diabetes Prediction Web App')
    
    # getting the input data from the user 
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose level')
    BloodPressure = st.text_input("Blood Pressure Value")
    Skinthickness = st.text_input("Skin Thickness Value")
    Insulinlevel = st.text_input("Insulin level")
    BMI = st.text_input('BMI')
    Diabetespedigreefunction = st.text_input('Diabetes Pedigree Function')
    Age = st.text_input('Age')
    
    # code for prediction 
    diagnosis = ''
    
    #creating a button for prediction 
    if st.button('Diabetes Test Result'):
        diagonosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,Skinthickness,Insulinlevel,BMI,Diabetespedigreefunction,Age])
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()