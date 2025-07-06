from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
import os
from pathlib import Path

ROOT = Path(__file__).parents[1].__str__()

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# Load model and encoders
model = joblib.load(os.path.join(ROOT, "models", "lgbm_model.joblib"))
encoders = joblib.load(os.path.join(ROOT, "models", "label_encoders.joblib"))

# Choices
CATEGORIES = ['Alkoholunfälle', 'Fluchtunfälle', 'Verkehrsunfälle'] 
ACCIDENT_TYPES = ['insgesamt', 'mit Personenschäden', 'Verletzte und Getötete']
MONTHS = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    return templates.TemplateResponse("form.html", {
        "request": request,
        "categories": CATEGORIES,
        "types": ACCIDENT_TYPES,
        "months": MONTHS
    })

@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    category: str = Form(...),
    acc_type: str = Form(...),
    year: int = Form(...),
    month: int = Form(...)
):
    # Validate year
    if not (1990 <= year <= 2050):
        return templates.TemplateResponse("form.html", {
            "request": request,
            "error": "Year must be between 1990 and 2050.",
            "categories": CATEGORIES,
            "types": ACCIDENT_TYPES,
            "months": MONTHS
        })

    # Encode categorical features
    try:
        category_encoded = encoders['Category'].transform([category])[0]
        type_encoded = encoders['Accident_type'].transform([acc_type])[0]
    except:
        return templates.TemplateResponse("form.html", {
            "request": request,
            "error": "Invalid category/type provided.",
            "categories": CATEGORIES,
            "types": ACCIDENT_TYPES,
            "months": MONTHS
        })

    input_array = np.array([[category_encoded, type_encoded, year, month]])
    prediction = model.predict(input_array)[0]
    prediction = round(prediction, 2)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "prediction": prediction,
        "year": year,
        "month": [k for k, v in MONTHS.items() if v == month][0],
        "category": category,
        "acc_type": acc_type
    })
