
import streamlit as st
import pickle

# Load Trained Model
with open("iris_model.pkl", "rb") as file:
    model = pickle.load(file)

# Project Title
st.title("🌸 Iris Flower Classification")

st.write("Enter Flower Measurements")

# User Input
sl = st.number_input("Sepal Length", 0.0, 10.0, 5.1)
sw = st.number_input("Sepal Width", 0.0, 10.0, 3.5)
pl = st.number_input("Petal Length", 0.0, 10.0, 1.4)
pw = st.number_input("Petal Width", 0.0, 10.0, 0.2)

# Prediction
if st.button("Predict Flower"):

    result = model.predict([[sl, sw, pl, pw]])

    names = ["Setosa", "Versicolor", "Virginica"]

    st.success("Predicted Flower: " + names[result[0]])
