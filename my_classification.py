import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def show_page(df):
    st.title("Logistic Regression (Classification)")
    X = df[['ai_replacement_score', 'skill_gap_index', 'skill_demand_growth_percent']]
    y = df['automation_risk_category']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    st.success(f"Classification Accuracy: {acc*100:.2f}%")