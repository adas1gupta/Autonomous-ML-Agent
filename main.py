import streamlit as st
import pandas as pd 
import json 
from dotenv import load_dotenv

st.title("AutoML Agent")
st.markdown("Upload a csv file to get started")

uploaded_file = st.file_uploader("Upload a csv file", type=["csv"])

def build_preprocessing_prompt(df):
    prompt = f"""
    You are an expert data scientist, specifically in the field of data cleaning. 
    You are given a dataframe, and you need to clean the data. 
    The data is as follows: 
    {df.head()}
    Please clean the data and return the cleaned data. 
    Make sure to handle the following:
    - Missing values
    - Duplicate values
    - Outliers

    {df.describe()}
    """
    return prompt 

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    selected_column = st.selectbox("Select a column to predict", 
    df.columns.tolist(), 
    help="The column to predict")

    run_button = st.button("Run AutoML")

    if button: 
        with st.spinner("Running AutoML..."):
            st.write(df.head())
            st.write(selected_column)
    
