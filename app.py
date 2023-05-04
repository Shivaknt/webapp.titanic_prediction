import streamlit as st
import numpy as np
from pickle import load

scaler = load(open('sc.pkl', 'rb'))
knn = load(open('knn.pkl', 'rb'))


st.header('Titanic Survival Prediction')



pclass =st.text_input("Pclass",placeholder='Enter the pclass value(1 or 2 or 3): ')
sex = st.text_input("Sex",placeholder='Enter the sex value(0 or 1): ')
age= st.text_input("Age",placeholder='Enter the age(like 20.0): ')
sibsp= st.text_input("Sibsp",placeholder='Enter the no of children: ')
parch=st.text_input("Parch",placeholder='Enter the val of parch(0 or 1): ')
fare=st.text_input("Fare",placeholder='Enter the fare value :')
embarked=st.text_input("Embarked",placeholder='Enter the embarked (0 or 1 or 2)')


btn_click = st.button("Predict")

if btn_click == True:
    if pclass and sex and age and sibsp and parch and fare and embarked:
        query_point = np.array([int(pclass),int(sex),float(age),int(sibsp),int(parch),float(fare),int(embarked)]).reshape(1, -1)
        query_point_transformed = scaler.transform(query_point)
        pred = knn.predict(query_point_transformed)
        if pred==1:
            st.success('Survived')
        else:
            st.success('Not Survived')
        #st.success(pred)
    else:
        st.error("Enter the values properly.")