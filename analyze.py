import pickle
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

def analyze_text(user_text):
    sentiment = model.predict([user_text])[0]
    confidence = model.predict_proba([user_text])[0][sentiment]
    sentiment_label = sentiment_map[sentiment]

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

    print(f"\n📝 Input: {user_text}")
    print(f"🔹 Sentiment: {sentiment_label} ({confidence:.2f} confidence)")
    print(f"💡 AI Suggestion: {ai_suggestion}\n")
if __name__ == "__main__":
    analyze_text("Customer service was terrible — nobody answered the phone.")
    analyze_text("Weather looks clear for our flight today.")
    analyze_text("Had a great experience flying with JetBlue today.")
