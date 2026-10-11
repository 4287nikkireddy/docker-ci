
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("ckd_dataset.csv")

target = "classification"

X = df.drop(columns=[target])
y = df[target]

# Convert categorical features into numeric columns
X = pd.get_dummies(X, dummy_na=True)
X = X.fillna(0)

model = make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)

joblib.dump(
    {"model": model, "columns": X.columns.tolist()},
    "ckd_model.pkl"
)

print("Model trained and saved as ckd_model.pkl")
print("Test accuracy:", model.score(X_test, y_test))
