# 📊 Professor Rating Predictor

This project uses data from [PlanetTerp](https://planetterp.com) to predict professor star ratings using GPA data, sentiment analysis from reviews, and machine learning models.

## 🔍 Overview

- Scrapes professor data (ratings, reviews, grades) via PlanetTerp API
- Computes:
  - Average GPA from grade distributions
  - Expected GPA from reviews
  - Sentiment scores using VADER
- Combines all data into a training dataset
- Trains four models:
  - Linear Regression
  - Random Forest
  - K-Nearest Neighbors
  - Support Vector Regression
- Evaluates models with 10-fold cross-validation

## 📂 Files

- `professors.json`: Raw professor data
- `professors_with_avg_gpa.json`: GPA computed from grades
- `professor_sentiment.json`: Sentiment scores
- `professor_features.csv`: Final training data
- Model evaluation output via console

## 🛠 Tech Used

- Python, Pandas, scikit-learn, VADER
- APIs: [PlanetTerp](https://planetterp.com/api)

## 📈 Results

All models evaluated with 10-fold CV. Best performance:

- **SVM**: MSE ≈ 0.50, R² ≈ 0.62
