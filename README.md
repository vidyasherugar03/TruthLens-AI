
# TruthLens AI 🔍

### Explainable Fake News Detection Using NLP and Machine Learning

## About the Project

TruthLens AI is a Machine Learning based web
application that analyzes news articles using
Natural Language Processing.

It predicts whether an article resembles fake
or real news based on patterns learned from
labelled training data.

The application also explains influential words
behind its predictions.

## Objectives

- Detect suspicious news language patterns.
- Apply NLP techniques to news articles.
- Train Machine Learning classification models.
- Explain model predictions.
- Build an interactive web application.

## Technologies

- Python
- Pandas and NumPy
- NLP
- TF-IDF
- Scikit-learn
- Streamlit
- Git and GitHub

## Project Status

Day 1: Project setup and GitHub integration.

## Disclaimer

TruthLens AI is an educational project.
Its predictions do not independently verify
the factual accuracy of news articles.

## Model Evaluation

TruthLens AI uses a Linear Support Vector Machine (SVM) model for fake-like and real-like news classification.

### Evaluation Metrics

| Metric | Score |
|---|---:|
| Accuracy | 95.01% |
| Precision | 94.65% |
| Recall | 94.07% |
| F1-Score | 94.36% |
| ROC-AUC | 98.84% |

### Evaluation Graphs

#### Confusion Matrix
![Confusion Matrix](screenshots/confusion_matrix.png)

#### ROC Curve
![ROC Curve](screenshots/roc_curve.png)

### Note
These results are based on the held-out test split of the WELFake dataset. Predictions indicate learned text patterns and do not establish whether a news article is factually true.
