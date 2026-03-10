import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder

def show_page(df):
    st.title("Data Preprocessing")
    
    st.write("### Label Encoding")
    le = LabelEncoder()
    # Encoding Categorical columns for the model
    df_encoded = df.copy()
    for col in ['industry', 'country', 'job_role']:
        df_encoded[col] = le.fit_transform(df[col])
    
    st.write("Encoded Data Preview:", df_encoded.head())

    st.write("### Feature Scaling")
    scaler = StandardScaler()
    num_cols = ['automation_risk_percent', 'salary_before_usd', 'skill_gap_index']
    df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])
    st.write("Standardized Values (Mean=0, Std=1):", df_encoded[num_cols].head())