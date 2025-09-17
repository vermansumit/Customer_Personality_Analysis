import logging
logging.basicConfig(filename = 'logs/train.log', 
                    level = logging.INFO, 
                    format = '%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting training run")

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from data_loader import load_data
from preprocessing import basic_clean

df = basic_clean(load_data())
target = 'Response'

# pick feature(example)
num_features = ['Income', 'Age', 'Recency', 'TotalSpend','NumWebVisitsMonth']
cat_features = ['Education', 'Marital_Status']

x = df[num_features + cat_features]
y = df[target].fillna(0).astype(int)

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

pipe = Pipeline([
    ('prep', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipe.fit(x_train, y_train)

#Evaluate
y_pred = pipe.predict(x_test)
print(classification_report(y_test, y_pred))
print("ROD AUC:", roc_auc_score(y_test, pipe.predict_proba(x_test)[:,1]))

# save model
joblib.dump(pipe, "models/random_forest_pipeline.joblib")

import mlflow
mlflow.set_experiment("customer_personality_experiment")

with mlflow.start_run():
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, pipe.predict_proba(x_test)[:,1]))
    mlflow.sklearn.log_model(pipe,"rf_pipeline")