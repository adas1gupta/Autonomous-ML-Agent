import streamlit as st
import pandas as pd 
import json 
from dotenv import load_dotenv
import io
import os
from openai import OpenAI

load_dotenv()

st.title("AutoML Agent")
st.markdown("Upload a csv file to get started")

uploaded_file = st.file_uploader("Upload a csv file", type=["csv"])

def summarize_dataset(dataframe: pd.DataFrame) -> str:
    """
    Generate a comprehensive summary of the dataset for LLM context. 

    This function creates a detailed text summary that includes: 
    - Column data types and schema information
    - Missing value counts and data completeness
    - Cardinality (unique value counts) for each column
    - Statistical summaries for numeric columns 
    - Sample data rows in CSV format
    
    Args: 
        dataframe: The pandas DataFrame to summarize

    Returns:
        A formatted string containing the dataset summary. 
    """
    try:
        buffer = io.StringIO()
        sample_rows = min(30, len(dataframe))
        dataframe.head(sample_rows).to_csv(buffer, index=False)
        sample_csv = buffer.getvalue()

        dtypes = dataframe.dtypes.astype(str).to_dict()
        non_null_counts = dataframe.notnull().sum().to_dict()
        null_counts = dataframe.isnull().sum().to_dict()
        nunique = dataframe.nunique(dropna=True).to_dict()

        numeric_cols = [c for c in dataframe.columns if pd.api.types.is_numeric_dtype(dataframe[c])]
        desc = dataframe[numeric_cols].describe().to_dict() if numeric_cols else {}

        lines = []

        lines.append("Schema (dtype):")
        for k, v in dtypes.item():
            lines.append(f"-{k}:{v}")
        lines.append("")

        lines.append("Cardinality (nunique):")
        for k, v in nunique.items():
            lines.append(f"- {k}: {int(v)}")
        lines.append("")

        lines.append("Sample rows (CSV head):")
        lines.append(sample_csv)

        return "\n".join(lines)

    except Exception as e:
        return f"Error summarizing dataset: {e}" 

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
    - Standardize the data accordingly 
    - Use one-hot-encoding for categorical variables

    Write a python script to clean the data, based on the data summary provided, in a json property called "script".
    The script should be a python script that can be executed to clean the data. 
    """
    return prompt 

def get_openai_script(prompt: str) -> str: 
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model="gpt-5-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system", "content": (
                        "You're a senior data engineer. Always return a strict JSON object matching the user's requested schema."
                    )
                },
                {
                    "role": "user", "content": prompt
                }
            ]
        )
        
        if not resp or not getattr(resp, 'choices', None):
            return None 
        
        text = resp.choices[0].message.content or ""

        try: 
            data = json.loads(text)
            script_val = data.get("script")
            if isinstance(script_val, str) and script_val.strip():
                return script_val.strip()
        except Exception as e:
            pass
        
    except Exception as e:
        return None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    selected_column = st.selectbox("Select a column to predict", 
    df.columns.tolist(), 
    help="The column to predict")

    run_button = st.button("Run AutoML")

    if run_button: 
        with st.spinner("Running AutoML..."):
            cleaning_prompt = build_preprocessing_prompt(df)
            with st.expander("Cleaning Prompt"):
                st.write(cleaning_prompt)
            script = get_openai_script(cleaning_prompt)
            with st.expander("Script"):
                st.code(script)    

