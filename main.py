import streamlit as st
import pandas as pd
import my_preprocessor as pre 
import my_regression as reg
import my_classification as cla

st.set_page_config(page_title="AI Job Impact Analysis", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv('ai_job_replacement_2020_2026_v2.csv')

df = load_data()

# 4. Setup Navigation
st.sidebar.title("ML Navigation")
page = st.sidebar.radio("Go to:", ["Data Overview", "Preprocessing", "Regression", "Classification"])

# 5. Use the 'page' and 'df' variables
if page == "Data Overview":
    st.title("AI Job Replacement Dataset")
    st.write(df.head()) # 'df' now exists!

elif page == "Preprocessing":
    pre.show_page(df) # Passing 'df' to your module

elif page == "Regression":
    reg.show_page(df)

elif page == "Classification":
    cla.show_page(df)