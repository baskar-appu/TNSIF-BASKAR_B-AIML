


import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


print("Loading dataset...")

df = pd.read_csv("spam.csv")

print("Dataset loaded successfully.")
print("Number of records:", len(df))



print("\nFirst 5 records:")
print(df.head())

# email = input text
# label = target
#
# label:
# 1 = Spam
# 0 = Not Spam

X = df["email"]
y = df["label"]



data = pd.DataFrame({
    "email": X,
    "label": y
})

print("\nClass distribution:")
print(y.value_counts())

print("\nClass meaning:")
print("0 = Not Spam")
print("1 = Spam")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
   
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=1000
        )
    )
])

print("\nTraining model...")

model.fit(X_train, y_train)

print("Model training completed.")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)


print("MODEL EVALUATION")


print(f"Accuracy: {accuracy * 100:.2f}%")


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Not Spam", "Spam"],
        zero_division=0
    )
)


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


joblib.dump(model, "spam_model.pkl")


print("MODEL SAVED")


print("Saved file: spam_model.pkl")


print("SAMPLE PREDICTIONS")


sample_emails = [
    "Congratulations! You have won a free iPhone. Click here to claim your prize.",
    "Hi, please send me the project report before tomorrow.",
    "URGENT! You have won $10000. Claim your money now.",
    "Can we meet tomorrow to discuss the project?"
]


predictions = model.predict(sample_emails)


for email, prediction in zip(sample_emails, predictions):

    if prediction == 1:
        result = "Spam"
    else:
        result = "Not Spam"

    print("\nEmail:", email)
    print("Prediction:", result)


print("\nTraining process completed successfully.")