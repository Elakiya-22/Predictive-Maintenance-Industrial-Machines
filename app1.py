import streamlit as st
import pickle
import numpy as np


# Load model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Predictive Maintenance of Industrial Machines")

air_temp = st.number_input("Air Temperature [K]", value=300.0)
process_temp = st.number_input("Process Temperature [K]", value=310.0)
rot_speed = st.number_input("Rotational Speed [rpm]", value=1500)
torque = st.number_input("Torque [Nm]", value=40.0)
tool_wear = st.number_input("Tool Wear [min]", value=10)

if st.button("Predict"):
    features = np.array([[air_temp, process_temp, rot_speed, torque, tool_wear]])

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("Machine Failure Predicted")
    else:
        st.success("Machine is Healthy")
