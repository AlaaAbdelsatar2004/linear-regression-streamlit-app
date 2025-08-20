
#pip install streamlit
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np 

from sklearn.linear_model  import LinearRegression
from sklearn.preprocessing  import  StandardScaler 

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error , mean_squared_error 

st.title("Linear Regression Application")
file = st.file_uploader("Please Upload Your File " , type=["csv"])
if file is not None:
    data = pd.read_csv(file)
    st.write("My Data Is ",data.head())
    st.write("The Describrion ",data.describe())
    features = st.multiselect("Select Features" , data.columns)
    target = st.multiselect("Select Target" , data.columns)

    # check missing data
    st.write("Check Missing Data",data.isna().sum())
    # check duplication
    st.write("Check Duplication",data.duplicated().sum())

    # check outliers
    st.write("Check Outliers")
    fig,ax = plt.subplots(figsize=(5,3))
    sns.boxenplot(data , ax=ax)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

    # split data
    xtrain , xtest , ytrain , ytest = train_test_split(data[features] , data[target] , test_size=0.2 , random_state=42)

    # scaling
    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.fit_transform(xtest)

    # modeling
    model = LinearRegression()
    model.fit(xtrain , ytrain)
    ypred = model.predict(xtest)

    # evaluation
    st.write("Evaluation")
    st.write("Mean Absolute Error", mean_absolute_error(ytest , ypred))
    st.write("Mean Squared Error", mean_squared_error(ytest , ypred))

    # user prediction data
    st.header("New Prediction")
    val={}
    for col in features:
        val[col]=st.number_input(f'{col}' , value=data[col].mean())
    st.write(val)
    input_data = pd.DataFrame([val])
    out = model.predict(input_data)
    st.write("The New Prediction = ",out)

