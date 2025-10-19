# 💬 Users Reviews Sentiment Analysis Project

This project builds and deploys an **interactive sentiment analysis dashboard** that analyzes user reviews of ChatGPT (or any similar product) and visualizes insights such as sentiment distribution, user satisfaction trends, and common feedback themes.

---

## 🚀 Project Overview

The project has two main components:

1. **Model Training (`nlpfinal.ipynb`)**
   - Preprocesses user review data.
   - Trains and evaluates multiple machine learning models for text sentiment classification.
   - Selects the best model (based on accuracy/F1 score).
   - Saves the trained model and vectorizer using `joblib` for later use in deployment.

2. **Streamlit Dashboard (`final.py`)**
   - Loads the trained sentiment model and dataset.
   - Provides an interactive UI to explore and visualize user sentiment trends.
   - Allows real-time sentiment prediction for new user inputs.
   - Generates word clouds, bar charts, and time-based sentiment trends.

---

## 🧠 Features

### 🧾 NLP Model (`nlpfinal.ipynb`)
- Text cleaning (stopword removal, lemmatization, tokenization).
- Vectorization using TF-IDF or CountVectorizer.
- Model comparison (e.g., Logistic Regression, Naive Bayes, SVM).
- Evaluation using accuracy, precision, recall, and F1 score.
- Exports:
  - `best_sentiment_model.pkl`
  - `best_sentiment_vectorizer.pkl`

### 📊 Dashboard (`final.py`)
- Built using **Streamlit** for interactive analysis.
- Sections include:
  1. **Try Your Own Review** – Input any review and get instant sentiment prediction.
  2. **Overall Sentiment Distribution** – Visual summary of sentiment balance.
  3. **Sentiment vs Rating** – Correlation between user ratings and sentiment.
  4. **Keywords per Sentiment** – Word clouds showing common terms for each sentiment.
  5. **Sentiment Over Time** – Monthly trends in user satisfaction.
  6. **Verified vs Non-Verified Users** – Comparison of verified user feedback.
  7. **Review Length vs Sentiment** – Relationship between review verbosity and sentiment.
  8. **Sentiment by Location** – Regional sentiment differences.
  9. **Sentiment by Platform/Version** – Variations by user platform or ChatGPT version.
  10. **Negative Feedback Themes** – Common phrases in negative reviews.

---

## 📁 Project Structure

```
├── final.py                      # Streamlit dashboard script
├── nlpfinal.ipynb                # Model training and evaluation notebook
├── reviews_cleaned.csv           # Preprocessed dataset (used by dashboard)
├── best_sentiment_model.pkl      # Trained sentiment classification model
├── best_sentiment_vectorizer.pkl # TF-IDF or CountVectorizer for feature extraction
└── README.md                     # Project documentation
```

---

## 🧩 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/sentiment-dashboard.git
cd sentiment-dashboard
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

*(If no requirements.txt, install manually:)*
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn wordcloud joblib
```

---

## ▶️ Usage

### Run the Streamlit App
```bash
streamlit run final.py
```

### Example Output
- Sentiment prediction for a sample review:
  ```
  Input: "ChatGPT is amazing! It helps me with everything."
  Output: Positive
  ```
- Dashboard sections will visualize trends dynamically based on dataset.

---

## 📊 Sample Insights
- Most user reviews are **positive**, indicating strong satisfaction.
- Negative reviews often mention themes like “slow response” or “inaccurate answers”.
- Review lengths vary — longer reviews often indicate more emotional engagement.

---

## 🧱 Tech Stack
| Component | Technology |
|------------|-------------|
| Language | Python 3 |
| Dashboard | Streamlit |
| NLP | Scikit-learn, TF-IDF |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Data Handling | Pandas, NumPy |

---

## 🧑‍💻 Author
**Your Name**
📧 your.email@example.com
🔗 GitHub / LinkedIn / Portfolio (optional)

---

## 📜 License
This project is licensed under the **MIT License** — you are free to use and modify it with attribution.
