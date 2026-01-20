import gradio as gr
import pandas as pd
import pickle 
import numpy as np

with open("Water _Potability_Model.pkl", "rb") as file:
    model = pickle.load(file)

def predwater(ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity):
   input_data = pd.DataFrame([[
         ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity
    ]], columns=[
         'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
    ])
   prediction = model.predict(input_data)[0]
   return "Potable" if prediction == 1 else "Not Potable"
   
input = [
    gr.Slider(0, 14 , label="pH Level"),
    gr.Number(label= "Hardness (mg/L)"),
    gr.Number(label= "Solids (ppm)"),
    gr.Number(label= "Chloramines (ppm)"),
    gr.Number(label= "Sulfate (mg/L)"),
    gr.Number(label= "Conductivity (μS/cm)"),
    gr.Number(label= "Organic Carbon (ppm)"),
    gr.Number(label= "Trihalomethanes (μg/L)"),
    gr.Number(label= "Turbidity (NTU)")
]



app= gr.Interface(
    fn=predwater,
    inputs=input,
    outputs=gr.Textbox(label="Water Potability"),
    title="Water Potability Predictor",
    description="Predict whether water is potable or not based on its chemical properties."
)

app.launch(share=True)

