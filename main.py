import streamlit as st
import pandas as pd
from openai import OpenAI 

st.title("Autonomous ML Agent")
st.header("Upload a file")

uploaded_dataset = st.file_uploader(label="dataset", type="csv", accept_multiple_files=False)

if uploaded_dataset is not None:
    df = pd.read_csv(uploaded_dataset)

    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} cols")
    st.dataframe(df.head(10))

    columns = [item for item in df.columns]
    target_col = st.selectbox("Column to predict", df.columns)

    client = OpenAI(LLM_KEY)

    prompt_text = """
    You are an expert Data Scientist. Given the dataset below, write Python code to process it. This involves handling null values, removing outliers, and standardizing values. 
    The column you will be predicting is {target_col}. 
    {df.describe()}
    df.head(10)

    """