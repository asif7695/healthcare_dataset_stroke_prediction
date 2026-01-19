import gradio as gr
import pandas as pd
import numpy as np
import pickle


with open("finalized_stroke_pred_model_with_tuned_lr.pkl",'rb') as file:
    model = pickle.load(file)

def predict_stroke(gender,age,hypertension,heart_disease,ever_married,work_type,residence_type,avg_glucose_level,bmi,smoking_status):
    if age <= 18:
        age_group = "young"
    elif age <= 35:
        age_group = "adult"
    elif age <= 55:
        age_group = "middle_aged"
    else:
        age_group = "Senior"

    input_df = pd.DataFrame([{
        "gender": gender,
        "age": age,
        "age_group": age_group, 
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status
    }])

    
    prediction = model.predict(input_df)[0]

    return "High Stroke Risk" if prediction == 1 else "Low Stroke Risk"
    


inputs = [
    gr.Dropdown(["Male","Female","Other"], label="Gender"),
    gr.Slider(1, 80, step=1, label="Age"),
    gr.Radio([0, 1], label="Hypertension"),
    gr.Radio([0, 1], label="Heart Disease"),
    gr.Radio(["Yes", "No"], label="Ever Married"),
    gr.Dropdown(["Private", "Self-employed", "Govt_job", "children", "Never_worked"],label="Work Type"),
    gr.Radio(["Urban", "Rural"], label="Residence Type"),
    gr.Slider(55.0, 170.0, step=0.1, label="Average Glucose Level"),
    gr.Slider(10.0, 50.0, step=0.1, label="BMI"),
    gr.Dropdown(["formerly smoked", "never smoked", "smokes", "Unknown"],label="Smoking Status")
]


app = gr.Interface(
    fn=predict_stroke,
    inputs=inputs,
    outputs="text",
)

# Launch
app.launch()
