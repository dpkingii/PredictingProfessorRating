import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

FEATURES = ["average_gpa", "average_expected_gpa", "sentiment"]

df = pd.read_csv("data/professor_features.csv").dropna()
X = df[FEATURES]
y = df["star_rating"]

# KNN and SVR are distance-based, so features need to be on the same scale
models = {
    "Mean baseline": DummyRegressor(),
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "KNN (k=23)": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=23)),
    "SVR (RBF)": make_pipeline(StandardScaler(), SVR(kernel="rbf")),
}

cv = KFold(n_splits=10, shuffle=True, random_state=42)
scoring = ["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"]

print(f"{len(df)} professors, 10-fold CV\n")
print(f"{'Model':<20}{'RMSE':>8}{'MAE':>8}{'R2':>8}")
for name, model in models.items():
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
    rmse = -scores["test_neg_root_mean_squared_error"].mean()
    mae = -scores["test_neg_mean_absolute_error"].mean()
    r2 = scores["test_r2"].mean()
    print(f"{name:<20}{rmse:>8.3f}{mae:>8.3f}{r2:>8.3f}")

sentiment_only = cross_validate(
    make_pipeline(StandardScaler(), SVR(kernel="rbf")), X[["sentiment"]], y, cv=cv, scoring="r2"
)
print(f"\nSVR on sentiment alone: R2 {sentiment_only['test_score'].mean():.3f}")
