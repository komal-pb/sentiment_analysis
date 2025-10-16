import streamlit as st
import pickle
from groq import Groq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Streamlit UI
st.title("✈️ Airline Sentiment Analyzer")

user_text = st.text_area("Enter text to analyze:", "")

if st.button("Analyze"):
    if user_text.strip() == "":
        st.warning("Please enter some text!")
    else:
        # Step 1: ML sentiment prediction
        sentiment = model.predict([user_text])[0]
        confidence = model.predict_proba([user_text])[0][sentiment]
        sentiment_label = sentiment_map[sentiment]

        # Step 2: Generate AI suggestion
        prompt = f"""
        The user said: "{user_text}".
        The detected sentiment is {sentiment_label}.
        Write a short, friendly emotional explanation and a helpful suggestion.
        Be empathetic and conversational.
        """
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=80,
        )
        ai_suggestion = completion.choices[0].message.content

        # Step 3: Display results
        st.markdown(f"**Sentiment:** {sentiment_label} ({confidence:.2f} confidence)")
        st.markdown(f"**AI Suggestion:** {ai_suggestion}")
