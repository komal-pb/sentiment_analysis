# Sentiment Analysis on Twitter (Sentiment140 Dataset)

## Project Overview
This project implements a **Sentiment Analysis model** to classify tweets as **positive** or **negative** based on their textual content.  
I used the **Sentiment140 dataset**, which contains **1.6 million labeled tweets**, and applied text preprocessing, feature extraction, and machine learning to build the model.  

---

## Features
- Data preprocessing: cleaning, stopword removal, stemming
- Text vectorization using **TF-IDF**
- Binary classification using **Logistic Regression**
- Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix
- Model saved with **Pickle** for reusability
- Predicts sentiment of new unseen tweets

---

## Dataset
- **Dataset used:** [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)  
- **Size:** 1.6 million tweets  
- **Classes:**  
  - `0` → Negative sentiment  
  - `4` → Positive sentiment (converted to `1` in preprocessing)  

---

## Technologies Used
- **Python**
- **Pandas, NumPy**
- **NLTK** (text preprocessing, stopwords, stemming)
- **Scikit-learn** (TF-IDF, Logistic Regression, evaluation metrics)
- **Jupyter Notebook**
- **Pickle** (saving and loading model)

---

## Model Performance
- **Training Accuracy:** ~79%  
- **Test Accuracy:** ~77%  


