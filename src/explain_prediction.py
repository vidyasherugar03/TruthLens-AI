
import joblib
import re

# Load the trained model and TF-IDF vectorizer
model = joblib.load("models/best_model.joblib")
tfidf = joblib.load("models/tfidf_vectorizer.joblib")

# Get important words and their model weights
feature_names = tfidf.get_feature_names_out()
coefficients = model.coef_[0]

def explain_prediction(news):

    # Clean the input news
    news = news.lower()
    news = re.sub(r"http\S+|www\S+", "", news)
    news = re.sub(r"[^a-z\s]", " ", news)
    news = re.sub(r"\s+", " ", news).strip()

    # Convert news into TF-IDF features
    news_vector = tfidf.transform([news])

    # Predict the label
    prediction = model.predict(news_vector)[0]

    # Get model decision score
    score = model.decision_function(news_vector)[0]

    if prediction == 0:
        result = "FAKE-LIKE NEWS"
        direction = -1
    else:
        result = "REAL-LIKE NEWS"
        direction = 1

    print("\nPrediction:", result)

    # Calculate each word's contribution
    contributions = news_vector.multiply(coefficients).toarray()[0]

    # Select words supporting the prediction
    important_indices = [
        i for i, value in enumerate(contributions)
        if value * direction > 0
    ]

    important_indices.sort(
        key=lambda i: abs(contributions[i]),
        reverse=True
    )

    print("\nImportant words influencing the prediction:")

    if not important_indices:
        print("No influential words found in the vocabulary.")
    else:
        for i in important_indices[:10]:
            print(
                f"{feature_names[i]} : "
                f"{abs(contributions[i]):.4f}"
            )

    print("\nNote: This is a model explanation, not factual verification.")

# Get news from the user
news = input("Enter a news headline or article: ")

explain_prediction(news)
