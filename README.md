# Heart Disease Risk Prediction Web App

A machine learning-powered web application that predicts the likelihood of heart disease based on user-entered medical attributes. Built with **Flask**, **Python**, and a **Random Forest Classifier**, this project combines data science, web development, and healthcare education.

## Project Overview

This web app allows users to input their health data—like age, cholesterol, chest pain type, etc.—and get a prediction about their heart disease risk. It also includes an information page to educate users about the medical relevance and safe ranges of each input field.

## Features

- Trained ML model using a Kaggle heart disease dataset
- User-friendly web form with real-time validation
- Educational info page for health awareness
- Clean, responsive, and interactive UI
- Model training using `scikit-learn` and `GridSearchCV`

## Tech Stack

- Python
- Flask
- scikit-learn
- Pandas
- HTML/CSS
- Bootstrap (for styling)

## Folder Structure

Heart-Disease-Predictor/
│
├── static/
│ ├── style.css
│ └── bg.jpg
│
├── templates/
│ ├── index.html
│ ├── result.html
│ └── about.html
│
├── heart.csv # Dataset
├── model.pkl # Trained model
├── train_model.py # ML model training script
├── app.py # Flask backend
└── README.md # Project documentation

## Installation

1. Clone the repository:
git clone https://github.com/samar-pratap-singh230/heart-disease-risk-predictor.git
cd heart-disease-risk-predictor
2. Create and activate a virtual environment:
python -m venv venv
venv\Scripts\activate # On Windows
3. Install dependencies:
pip install -r requirements.txt
4. Train the model:
python train_model.py
5. Run the app:
python app.py
6. Open your browser and visit:
http://127.0.0.1:5000/

## Sample Inputs

| Field             | Example Value |
|------------------|---------------|
| Age              | 52            |
| Sex              | M             |
| ChestPainType    | ATA           |
| RestingBP        | 130           |
| Cholesterol      | 250           |
| FastingBS        | 0             |
| RestingECG       | Normal        |
| MaxHR            | 165           |
| ExerciseAngina   | N             |
| Oldpeak          | 1.2           |
| ST_Slope         | Up            |

## Dataset

- Source: [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- Contains medical records and diagnosis labels (HeartDisease: 0 or 1)

## License

This project is open-source and available under the [MIT License](LICENSE).

---

**Developed by [Samar Pratap Singh]** — elevating awareness through data and design.
