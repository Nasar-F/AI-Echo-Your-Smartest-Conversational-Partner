# ==================================================
# 🧠 ChatGPT Reviews Sentiment Analysis Dashboard
# ==================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pickle
import joblib
from datetime import datetime

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
#st.set_page_config(page_title="Reviews Sentiment Dashboard", layout="wide")
#st.title("💬 User Review Sentiment Analyzer")
#st.markdown("Analyze the sentiment of user reviews. (Positive / Neutral / Negative).")
# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(page_title="Reviews Sentiment Dashboard", layout="wide")

# Centered Title and Subtitle
st.markdown(
    """
    <h1 style='text-align: center;'>💬 User Reviews Sentiment Analyzer</h1>
    <p style='text-align: center;'>Analyzes the sentiment of user reviews (Positive / Neutral / Negative).</p>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Load Model and Data
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("reviews_cleaned.csv")
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df

@st.cache_resource
def load_models():
    model = joblib.load("best_sentiment_model.pkl")
    vectorizer = joblib.load("best_sentiment_vectorizer.pkl")
    return model, vectorizer

df = load_data()
model, vectorizer = load_models()

# --------------------------------------------------
# Prediction Function
# --------------------------------------------------
def predict_sentiment(text):
    if not isinstance(text, str) or text.strip() == "":
        return "Neutral"
    try:
        X = vectorizer.transform([text])
        pred = model.predict(X)[0]
        return pred
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return "Error"

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("🧭 Navigation")
section = st.sidebar.radio(
    "Select Section:",
    [
        "Try Your Own Review",
        "1️⃣ Overall Sentiment Distribution",
        "2️⃣ Sentiment vs Rating",
        "3️⃣ Keywords per Sentiment",
        "4️⃣ Sentiment Over Time",
        "5️⃣ Verified vs Non-Verified",
        "6️⃣ Review Length vs Sentiment",
        "7️⃣ Sentiment by Location",
        "8️⃣ Sentiment by Platform",
        "9️⃣ Sentiment by Version",
        "🔟 Negative Feedback Themes",
    ]
)

# --------------------------------------------------
# 🧠 User Input Sentiment Prediction
# --------------------------------------------------
if section == "Try Your Own Review":
    st.title("Try Your Own Review")
    user_input = st.text_area("Enter your review for prediction:", height=150, placeholder="Type or paste a user review here...")

    if st.button("🔍 Analyze Sentiment"):
        if user_input.strip():
            sentiment = predict_sentiment(user_input)
            st.metric("Predicted Sentiment", sentiment)
        else:
            st.warning("Please enter some text to analyze.")

# --------------------------------------------------
# Section 1: Overall Sentiment Distribution
# --------------------------------------------------
if section == "1️⃣ Overall Sentiment Distribution":
    st.title("📊 Overall Sentiment of Reviews")

    if 'sentiment' not in df.columns:
        df['sentiment'] = df['clean_review'].apply(predict_sentiment)

    sentiment_counts = df['sentiment'].value_counts()
    st.bar_chart(sentiment_counts)
    st.write("Insight: Distribution shows overall customer satisfaction patterns.")

# --------------------------------------------------
# Section 2: Sentiment vs Rating
# --------------------------------------------------
if section == "2️⃣ Sentiment vs Rating":
    st.title("⭐ Sentiment vs Rating Analysis")

    if 'rating' in df.columns:
        # Generate sentiment if missing
        if 'sentiment' not in df.columns:
            df['sentiment'] = df['clean_review'].apply(predict_sentiment)

        # Create crosstab
        cross_tab = pd.crosstab(df['rating'], df['sentiment'])

        # Show as bar chart
        st.bar_chart(cross_tab)

        st.write("This shows how review sentiments vary with numeric ratings.")
    else:
        st.warning("The dataset does not contain a 'rating' column.")


# --------------------------------------------------
# Section 3: Keywords per Sentiment (Simple)
# --------------------------------------------------
if section == "3️⃣ Keywords per Sentiment":
    st.title("🔠 Keywords or Phrases per Sentiment")

    if "sentiment" not in df.columns:
        df["sentiment"] = df["clean_review"].astype(str).apply(predict_sentiment)

    sentiments = ["Positive", "Neutral", "Negative"]

    for s in sentiments:
        st.subheader(f"{s} Reviews")
        text = " ".join(df[df["sentiment"] == s]["clean_review"].astype(str))

        if text.strip():
            wc = WordCloud(width=800, height=400, background_color="white").generate(text)
            st.image(wc.to_array(), use_column_width=True)
        else:
            st.write(f"No {s.lower()} reviews available.")


# --------------------------------------------------
# Section 4: Sentiment Over Time (Simple)
# --------------------------------------------------
if section == "4️⃣ Sentiment Over Time":
    st.title("📆 Sentiment Trend Over Time")

    # Ensure sentiment column exists
    if "sentiment" not in df.columns:
        df["sentiment"] = df["clean_review"].astype(str).apply(predict_sentiment)

    # Ensure date column is in datetime format
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Drop rows with missing dates
        df = df.dropna(subset=["date"])

        # Group by month and sentiment
        trend = (
            df.groupby([pd.Grouper(key="date", freq="M"), "sentiment"])
            .size()
            .unstack(fill_value=0)
        )

        # Line chart
        st.line_chart(trend)
        st.write("Insight: Observe how positive, neutral, and negative sentiments fluctuate over time.")
    else:
        st.warning("⚠️ No 'date' column found in the dataset.")


# --------------------------------------------------
# Section 5: Verified vs Non-Verified
# --------------------------------------------------
if section == "5️⃣ Verified vs Non-Verified":
    st.title("✅ Verified vs ❌ Non-Verified Users")

    if "verified_purchase" in df.columns:
        # Ensure sentiment exists — predict if missing
        if "sentiment" not in df.columns:
            df["sentiment"] = df["clean_review"].apply(predict_sentiment)

        # Create simple count comparison
        counts = pd.crosstab(df["verified_purchase"], df["sentiment"])
        st.bar_chart(counts)
        st.write("Insight: Verified users tend to leave more positive or negative reviews?")
    else:
        st.warning("Column 'verified_purchase' not found in dataset.")


# --------------------------------------------------
# Section 6: Review Length vs Sentiment (Very Simple)
# --------------------------------------------------
if section == "6️⃣ Review Length vs Sentiment":
    st.title("✍️ Review Length vs Sentiment")

    # Make sure we have reviews
    if "clean_review" not in df.columns:
        st.warning("No 'clean_review' column found.")
    else:
        # Predict sentiment if not already done
        if "sentiment" not in df.columns:
            df["sentiment"] = df["clean_review"].astype(str).apply(predict_sentiment)

        # Calculate review length
        df["review_length"] = df["clean_review"].astype(str).apply(lambda x: len(x.split()))

        # Show average length by sentiment
        avg_len = df.groupby("sentiment")["review_length"].mean()

        st.bar_chart(avg_len)
        st.write("✅ Longer bars indicate which sentiment tends to have longer reviews.")


# --------------------------------------------------
# Section 7: Sentiment by Location (Simple Version)
# --------------------------------------------------
if section == "7️⃣ Sentiment by Location":
    st.title("🌍 Sentiment by Location")

    # Check if 'location' column exists
    if "location" not in df.columns:
        st.warning("No 'location' column found in the dataset.")
    else:
        # Predict sentiment if not already available
        if "sentiment" not in df.columns:
            df["sentiment"] = df["clean_review"].astype(str).apply(predict_sentiment)

        # Group by location and sentiment
        loc_sent = df.groupby(["location", "sentiment"]).size().unstack(fill_value=0)

        # Show top 10 locations with most reviews
        top_locs = loc_sent.sum(axis=1).sort_values(ascending=False).head(10)
        loc_sent_top = loc_sent.loc[top_locs.index]

        st.bar_chart(loc_sent_top)
        st.write("✅ Shows which locations have more positive or negative reviews.")

# --------------------------------------------------
# Section 8: Sentiment by Platform (Simple Version)
# --------------------------------------------------
if section == "8️⃣ Sentiment by Platform":
    st.title("🧑‍💻 Sentiment by Platform ")

    # Check if 'platform' column exists
    if "platform" not in df.columns:
        st.warning("No 'platform' column found in the dataset.")
    else:
        # Predict sentiment if not already done
        if "sentiment" not in df.columns:
            df["sentiment"] = df["clean_review"].astype(str).apply(predict_sentiment)

        # Group by platform and sentiment
        plat_sent = df.groupby(["platform", "sentiment"]).size().unstack(fill_value=0)

        # Show bar chart
        st.bar_chart(plat_sent)
        st.write("✅ Compare user sentiment between Web and Mobile platforms.")


# --------------------------------------------------
# Section 9: Sentiment by ChatGPT Version (Simple)
# --------------------------------------------------
if section == "9️⃣ Sentiment by Version":
    st.title("🤖 Sentiment by Version")

    # Check if 'version' column exists
    if "version" not in df.columns:
        st.warning("No 'version' column found in the dataset.")
    else:
        # Predict sentiment if not already done
        if "sentiment" not in df.columns:
            df["sentiment"] = df["clean_review"].astype(str).apply(predict_sentiment)

        # Group by version and sentiment
        version_sent = df.groupby(["version", "sentiment"]).size().unstack(fill_value=0)

        # Display as bar chart
        st.bar_chart(version_sent)
        st.write("✅ Compare how user sentiment changes across different versions.")


# --------------------------------------------------
# Section 10: Common Negative Feedback Themes (with Phrases)
# --------------------------------------------------
from sklearn.feature_extraction.text import CountVectorizer

if section == "🔟 Negative Feedback Themes":
    st.title("❌ Common Negative Feedback Themes")

    # Predict sentiment if not already available
    if "sentiment" not in df.columns:
        df["sentiment"] = df["clean_review"].astype(str).apply(predict_sentiment)

    # Filter only negative reviews
    negative_reviews = df[df["sentiment"] == "Negative"]

    if negative_reviews.empty:
        st.warning("No negative reviews found in the dataset.")
    else:
        #st.subheader("🧾 Most Common Negative Feedback Phrases")

        # Extract frequent phrases (bigrams and trigrams)
        vectorizer = CountVectorizer(
            stop_words='english',
            ngram_range=(2, 3),     # 2 = bigrams, 3 = trigrams
            max_features=15
        )
        word_matrix = vectorizer.fit_transform(negative_reviews["clean_review"].astype(str))
        word_counts = pd.DataFrame({
            "Phrase": vectorizer.get_feature_names_out(),
            "Frequency": word_matrix.toarray().sum(axis=0)
        }).sort_values(by="Frequency", ascending=False)

        # Show as table and chart
        #st.dataframe(word_counts)
        st.bar_chart(word_counts.set_index("Phrase"))

        st.write("💡 These phrases represent the most common complaint themes or pain points mentioned by users.")
