# Professor Rating Predictor

This project uses data from [PlanetTerp](https://planetterp.com) to predict UMD professor star ratings from grade data, the grades students expect to get, and sentiment analysis of their reviews.

## Results

36,733 reviews of 4,211 professors were collected. 3,658 professors had complete data and were evaluated with 10-fold cross-validation on the full dataset. Ratings are on a 1-5 scale.

| Model | RMSE | MAE | R² |
| --- | --- | --- | --- |
| Mean baseline | 1.096 | 0.910 | 0.00 |
| Linear Regression | 0.678 | 0.489 | 0.61 |
| Random Forest | 0.699 | 0.500 | 0.59 |
| KNN (k=23) | **0.667** | 0.479 | **0.63** |
| SVR (RBF) | 0.671 | **0.451** | 0.62 |

- KNN cuts RMSE by 39% compared to always guessing the average rating.
- The models all land within 0.03 R² of each other, and plain linear regression is almost as good as the rest. Most of the signal is linear.
- Sentiment does most of the work. SVR on sentiment alone gets R² 0.57, and the random forest gives it 74% of its feature importance.

## How it works

1. Pull every professor with a rating, along with their reviews, from the PlanetTerp API.
2. Build three features per professor:
   - **Average GPA**, computed from their historical grade distributions
   - **Average expected GPA**, from the grade each reviewer said they expected
   - **Average sentiment**, the VADER compound score over their review text
3. Join the features with each professor's star rating and drop professors with missing data.
4. Train and compare four regression models.

## Running it

```
pip install -r requirements.txt
python evaluate.py
```

`evaluate.py` reproduces the table above from the cached data in `data/`, so it doesn't need to call the API. `Predicting_Rating_using_ML.ipynb` has the full pipeline, including scraping and feature building.

## Files

- `data/professors.json`: raw professor data and reviews
- `data/professors_with_avg_gpa.json`: GPA computed from grades
- `data/professor_gpa_features.json`: average GPA and expected GPA per professor
- `data/professor_sentiment.json`: sentiment scores
- `data/professor_features.csv`: final training data
- `data/train_reviews.json`, `data/test_reviews.json`: review-level text and ratings
- `slide_deck.pdf`: project presentation

## Limitations

- Sentiment and expected GPA come from the same reviews that produce the star rating. The model is really measuring how well review text explains a rating, not predicting a rating before anyone reviews the professor.
- The median professor has only 3 reviews, so many target ratings are noisy.
- VADER is a general-purpose sentiment lexicon and misses course-specific language like "tough grader".

## Tech Used

- Python, pandas, scikit-learn, VADER
- APIs: [PlanetTerp](https://planetterp.com/api)
