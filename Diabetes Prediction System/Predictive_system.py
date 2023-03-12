# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle 

loaded_model = pickle.load(open("C:/Users/17347/Sapna's Projects/Diabetes Prediction System/diabetes_trained_model.sav",'rb'))
input_data = (4,110,92,0,0,37.6,0.191,30)

#changing the input_data to numpy array 
input_data_processed = np.asarray(input_data)
#reshape the array as we are predicting only for one instance out of all the 768 samples
input_data_reshaped = input_data_processed.reshape(1,-1)

# std_data = scaler.transform(input_data_reshaped)
# print(std_data)
# we are not using the standard scaler function here

prediction = loaded_model.predict(input_data_reshaped)
if prediction[0] == 0:
    print("Patient is Non-Diabetic")
else:
    print("Patient is Diabetic")