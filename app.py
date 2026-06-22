import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open('ipl_model.pkl', 'rb'))
features = ['runs_left', 'balls_left', 'wickets_left', 'rrr']

st.title("🏏 IPL Win Predictor")

batting_team = st.text_input("Batting Team Name")
target       = st.number_input("Target Score", min_value=1)
runs_left    = st.number_input("Runs Left", min_value=0)
balls_left   = st.number_input("Balls Left", min_value=1)
wickets_left = st.number_input("Wickets Left", min_value=0, max_value=10)

if st.button("Predict"):
    rrr = (runs_left / balls_left) * 6
    input_df = pd.DataFrame([[runs_left, balls_left, wickets_left, rrr]], columns=features)
    prob = model.predict_proba(input_df)[0]
    
    st.success(f"✅ Win Probability: {round(prob[1]*100, 2)}%")
    st.error(f"❌ Lose Probability: {round(prob[0]*100, 2)}%")