import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def show_page(df):
    st.title("Advanced Salary Prediction")

    X = df[['salary_before_usd', 'automation_risk_percent', 'ai_replacement_score', 'ai_adoption_level']]
    y = df['salary_after_usd']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    pipe.fit(X_train, y_train)

    predictions = pipe.predict(X_test)
    score = r2_score(y_test, predictions)

    st.success(f"Model R² Score: {score:.2f}")
    st.write("An R² score closer to 1.0 means the model is very accurate.")