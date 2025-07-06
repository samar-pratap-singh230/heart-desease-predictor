import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib

# Load dataset
heart_data = pd.read_csv("heart.csv")

X = heart_data.drop("HeartDisease", axis=1)
y = heart_data["HeartDisease"]

# Preprocessing
numeric = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
categorical = ["Sex", "ChestPainType", "FastingBS", "RestingECG", "ExerciseAngina", "ST_Slope"]

numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric),
    ("cat", categorical_transformer, categorical)
])

pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                           ("model", RandomForestClassifier(random_state=42))])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Grid search
params = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20]
}

grid = GridSearchCV(pipeline, params, cv=5, scoring="roc_auc")
grid.fit(X_train, y_train)

print("Best AUC:", grid.best_score_)

# Save model
joblib.dump(grid.best_estimator_, "model.pkl")