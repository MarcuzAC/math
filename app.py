from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.staticfiles import StaticFiles  # Import StaticFiles
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load the pre-trained model
model_path = 'models/gradient_model2.pkl'
scaler_path = 'models/scaler2.pkl'

if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
else:
    model = None
    scaler = None

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Route to serve the HTML form
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction route
@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    Mother_edu: int = Form(...),
    Father_edu: int = Form(...),
    Peerpractice: int = Form(...),
    studytime: int = Form(...),
    famsup: int = Form(...),
    Affordfees: int = Form(...),
    goals: int = Form(...),
    Mathteachers: int = Form(...),
    XulResources: int = Form(...),
    EnglishScore: int = Form(...),
    teachingmethod_Lecture: int = Form(...)
):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded properly")

    input_data = {
        "Mother_edu": Mother_edu,
        "Father_edu": Father_edu,
        "Peerpractice": Peerpractice,
        "studytime": studytime,
        "famsup": famsup,
        "Affordfees": Affordfees,
        "goals": goals,
        "Mathteachers": Mathteachers,
        "XulResources": XulResources,
        "EnglishScore": EnglishScore,
        "teachingmethod_Lecture": teachingmethod_Lecture
    }
    print("Received input data:", input_data)

    try:
        # Create a pandas DataFrame with appropriate columns
        features_df = pd.DataFrame([input_data])

        # Scale the input data using the scaler
        scaled_input = scaler.transform(features_df)

        # Make the prediction
        prediction = model.predict(scaled_input)
        predicted_class = int(prediction[0])
        print("Prediction result:", predicted_class)
    except Exception as e:
        print("Error in prediction:", e)
        raise HTTPException(status_code=500, detail="Prediction failed")

    recommendations = generate_recommendations(input_data)
     # Pass prediction and recommendations to the template for rendering
    return templates.TemplateResponse("results.html", {
        "request": request,
        "result_data": {"prediction": predicted_class, "recommendations": recommendations}
    })

# Function to generate study recommendations
def generate_recommendations(input_data):
    recommendations = []

    # Study time recommendation based on condition
    if input_data.get('studytime', 0) < 1:
        recommendations.append("Improve on studying time.")
    else:
        recommendations.append("Student shows good effort in individual studies.")

    # Teaching method recommendation
    if input_data.get('teachingmethod_Lecture', 0) == 1:
        recommendations.append("Use Interactive method for teaching.")
    else:
        recommendations.append("Lecture method appears to be fine for the student.")

    # English score recommendation
    if input_data.get('EnglishScore', 0) < 50:
        recommendations.append("Engage the Student's understanding of English.")
    else:
        recommendations.append("Student has a good understanding of English.")

    # Peer practice recommendation
    if input_data.get('Peerpractice', 0) == 1:
        recommendations.append("Recommend Student to Math Groups.")
    else:
        recommendations.append("Encourage the student to study independently or with a peer group.")


    return recommendations


    #return JSONResponse({
     #   'prediction': predicted_class,
      #  'recommendations': recommendations
    #})

if __name__ == "__main__":
    app.run(debug=True)
