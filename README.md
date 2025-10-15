# ✈️ Airline Sentiment Analyzer with AI Suggestions

A web application that predicts the sentiment of airline-related text (tweets, reviews, etc.) and provides a friendly, AI-generated suggestion for the user. Built using **Python**, **Streamlit**, **Scikit-learn**, and **Groq LLM API**.  

---

## 🚀 Features

- Predicts sentiment: **Positive**, **Neutral**, or **Negative**  
- Shows **confidence score** of prediction  
- Generates **friendly AI suggestions** based on detected sentiment  
- Simple and interactive **web interface** using Streamlit  
- Fully deployed and accessible online  

---

## 🧰 Technologies Used

- **Python 3.12+**  
- **Streamlit** – Frontend UI  
- **Scikit-learn** – Machine learning pipeline (TF-IDF + Logistic Regression)  
- **Groq LLM API** – AI-powered suggestions  
- **Pickle** – Saving/loading trained ML model  
- **Pandas & Numpy** – Data processing  
- **NLTK** – Text preprocessing (stopwords, stemming)  

---

## Model Performance

- **Accuracy on training data:** 89%
- **Accuracy on test data:** 85%

---

## Usage

- Enter text (e.g., airline review or tweet) in the input box
- Click Analyze
- View predicted sentiment, confidence, and AI-generated suggestion

---

## Screenshots

<img width="1905" height="998" alt="image" src="https://github.com/user-attachments/assets/03b43987-14f2-49af-81b7-e8d94a7337bd" />

---

## Future Improvements

- Integrate with live Twitter API to analyze tweets in real-time
- Add more ML models (Random Forest, SVM) for better prediction
- Improve frontend UI with charts and visualizations
