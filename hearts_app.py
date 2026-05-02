import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import StandardScaler,LabelEncoder
import pickle
import matplotlib.pyplot as plt



data=pd.read_csv('cleaned_heartdisease_data.csv')
df=pickle.load(open('data.pkl','rb'))

le_sex=LabelEncoder()
le_chest=LabelEncoder()
le_st=LabelEncoder()
le_ex=LabelEncoder()

df['Sex'] = le_sex.fit_transform(df['Sex'])
df['ChestPainType'] = le_chest.fit_transform(df['ChestPainType'])
df['ST_Slope'] = le_st.fit_transform(df['ST_Slope'])
df['ExerciseAngina'] = le_ex.fit_transform(df['ExerciseAngina'])



x=df.drop('HeartDisease',axis=1)
y=df['HeartDisease']

scaler=StandardScaler()
x_scaler=scaler.fit_transform(x)


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=20,random_state=42)
model=model = LogisticRegression(
    C=1,
    penalty='l2',
    solver='liblinear',
    class_weight='balanced'
)
model.fit(x_train,y_train)


st.title('Heart Disease Predictions:')
st.write('Apna Health parameters dale:')
st.write(data)
st.sidebar.header('Patient Data')

# Load model



age=st.sidebar.slider('Age',20,80,50)
sex=st.sidebar.selectbox('Sex',['M','F'])
cp=st.sidebar.selectbox('ChestPainType',['ASY','NAP','ATA','TA'])
maxhr=st.sidebar.number_input('MaxHR',min_value=50,max_value=210,value=70)
ea=st.sidebar.selectbox('ExerciseAngina',['N','Y'])
oldpeak=st.sidebar.number_input('OldPeak',min_value=-3.0,max_value=7.0,value=0.0)       
stslope=st.sidebar.selectbox('ST_Slope',['Flat','Up','Down'])
       


sex=le_sex.transform([sex])[0]
cp=le_chest.transform([cp])[0]
ea=le_ex.transform([ea])[0]
stslope=le_st.transform([stslope])[0]


input_data=[[age,sex,cp,maxhr,ea,oldpeak,stslope]]

input_scaler=scaler.transform(input_data)

if st.button('predict heart disease:'):
    prediction = model.predict(input_scaler)
    prob = model.predict_proba(input_scaler)

    st.success(f'Prediction Heart Disease: {prediction[0]}')

    # 🔹 Probability Chart (Model Behavior)
    st.subheader("Model Prediction Confidence")

    labels = ["No Disease", "Disease"]
    values = prob[0]   # [prob_no, prob_yes]

    x = [0, 1]

    fig, ax = plt.subplots()

    ax.plot(x, values, marker='o', linestyle='-', linewidth=2)
    ax.fill_between(x, values, alpha=0.2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylabel("Probability")
    ax.set_xlabel("Class")
    ax.set_title("Prediction Confidence")

    st.pyplot(fig)