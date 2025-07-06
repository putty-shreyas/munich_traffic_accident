# Accident Prediction App – AI Engineer Challenge

An AI-powered web application that predicts the number of road traffic accidents in Munich based on historical patterns. Built with a lightweight LightGBM model, the application provides fast and reliable predictions through a clean and user-friendly FastAPI interface.

---

## ✨ Features

- 📊 **Data Visualization:** Visual plots of accident trends from 2000–2020  
- 🧠 **ML Model:** Optimized `LGBMRegressor` trained with `RandomizedSearchCV`  
- 📈 **Evaluation Metrics:** MAE and RMSE calculated for model performance  
- 🌐 **Web App:** Built using `FastAPI`, enhanced with HTML + Tailwind UI  
- ☁️ **Deployment:** Live on [Render](https://render.com/)

---

## 🖼️ Visual Trends

<div align="center">
    <img src="assets/plot_example.png" alt="Accident Trends Plot" width="600" />
    <p>Accident Trends from 2000–2020</p>
</div>

---

## 🔍 How the App Works

- Users can select:
  - Accident **Category** (e.g. Alkoholunfälle *(Alcohol-related accidents)*)
  - Accident **Type** (e.g. Insgesamt *(Total)*)
  - Input **Year** and **Month**
- The model encodes inputs and returns the predicted number of accidents.

---

## ✅ Model Performance

- **Mean Absolute Error (MAE):** 55.26  
- **Root Mean Squared Error (RMSE):** 107.78  
- **Model Type:** LightGBM  
- **Best Parameters:** Optimized using Randomized Search with `n_iter=50`

---

## 🚀 Getting Started

 - Clone the repository and create a virtual environment.
 - Download the required dependencies and run the run_all.py file in the environment.
```
run run_all.py
```

---

## Live App

Access the deployed version of the app at 

---

## Tech Stack

- Python, Pandas, scikit-learn, LightGBM

- FastAPI, Jinja2, HTML, Tailwind CSS

- Render (Deployment), Git, GitHub

---

## About Me
I am Shreyas Putty, a M.Sc. Graduate in Data Science and Machine Learning and I am passionate about finding creative solutions through my knowledge and skills. I have 3+ years of experience in Python and am open to any new opportunities.

---

## Contact
We can connect through my email id - putty.shreyas@gmail.com and through my Linkedin - https://www.linkedin.com/in/shreyas-subhash-putty/