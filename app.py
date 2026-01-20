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
   
input=[
    gr.Slider(0,14,step=0.1,label="ph"),
    gr.Number(label="Hardness"),
    gr.Number(label="Solids"),
    gr.Number(label="Chloramines"),
    gr.Number(label="Sulfate"),
    gr.Number(label="Conductivity"),
    gr.Number(label="Organic_carbon"),
    gr.Number(label="Trihalomethanes"),
    gr.Number(label="Turbidity")
]



app= gr.Interface(
    fn=predwater,
    inputs=input,
    outputs=gr.Textbox(label="Water Potability"),
    title="Water Potability Predictor"
)

app.launch(share=True)

