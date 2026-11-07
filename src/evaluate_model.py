
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

# Load saved model and testing data
model = joblib.load("models/best_model.joblib")
data = joblib.load("models/tfidf_data.joblib")

X_test = data["X_test"]
y_test = data["y_test"]

# Generate predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics
print("\nTRUTHLENS AI - MODEL EVALUATION")
print("--------------------------------")

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall:", round(recall_score(y_test, y_pred), 4))
print("F1 Score:", round(f1_score(y_test, y_pred), 4))

# Calculate ROC-AUC using decision scores
y_scores = model.decision_function(X_test)
print("ROC-AUC:", round(roc_auc_score(y_test, y_scores), 4))

# Classification report
print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))

# Create screenshots folder
os.makedirs("screenshots", exist_ok=True)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Fake-like", "Real-like"]
)

display.plot()
plt.title("TruthLens AI - Confusion Matrix")
plt.savefig("screenshots/confusion_matrix.png")
plt.show()
plt.close()

# ROC curve
RocCurveDisplay.from_predictions(y_test, y_scores)
plt.title("TruthLens AI - ROC Curve")
plt.savefig("screenshots/roc_curve.png")
plt.show()
plt.close()

print("\nEvaluation graphs saved in screenshots folder!")
