# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 23:35:01 2024

@author: Vaidik
"""

import numpy as np
import pickle
import streamlit as st

# loading the model
model = pickle.load(open("C:\/Users/Vaidik/OneDrive/Desktop/ML Models to deploy/Diabetes_Model/clf.pkl", 'rb')) #path of pickle file

# creating the function for prediction
def diabetes_prediction(input_data):

    # changing the data to numpy array

    input_data_as_array = np.asarray(input_data)

    # reshaping the array

    input_data_as_array = input_data_as_array.reshape(1, -1)


    prediction = model.predict(input_data_as_array)
    print(prediction[0])

    if prediction[0] == 0:
        return "The person is non diabetic"

    else :
        return "The person is diabetic"
    
def main():
    
    # giving a title
    st.title("Diabetes Prediction Web App")
    
    # giving devloper name
    st.header("By Vaidik")
    
    # getting a input data from the user
    
    Pregnancies = st.text_input("Number_of_pregnancies")
    Glucose  = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure Value")
    SkinThickness = st.text_input("Skin Thickness Value")
    Insulin = st.text_input("Insulin Level")
    BMI = st.text_input("BMI Value")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value")
    Age = st.text_input("Age of the person")
    
    # code for prediction
    
    diagnosis = ''
    
    # creating a button for prediction
    
    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
    st.success(diagnosis)
        
if __name__ == "__main__":
    main()