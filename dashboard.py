"""
dashboard.py
------------
Interactive Social Media Sentiment Dashboard (Streamlit app).

Lets the user:
  - Upload their own CSV of reviews/posts/tweets, or use the bundled
    sample dataset
  - Pick which column holds the text
  - See overall sentiment distribution (pie + counts)
  - See a sentiment-over-time trend (if a date column is available)
  - Generate word clouds for positive vs. negative posts
  - Browse/search the row-level results table
  - Download the analyzed results as CSV
  - Try live "type your own text" sentiment check

Run locally:
    streamlit run dashboard.py

Deploy online (free):
    Push this project to a public GitHub repo, then deploy at
    https://share.streamlit.io (Streamlit Community Cloud) pointing it
    at dashboard.py. No server management needed.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from sentiment_analyzer import analyze_dataframe, analyze_text, sentiment_summary

st.set_page_config(page_title="Social Media Sentiment Dashboard", layout="wide")

SENTIMENT_COLORS = {"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#95a5a6"}


@st.cache_data
def load_sample_data():
    return pd.read_csv("sample_reviews.csv")


def render_pie_chart(counts: pd.Series):
    fig, ax = plt.subplots()
    colors = [SENTIMENT_COLORS.get(label, "#3498db") for label in counts.index]
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
           colors=colors, startangle=90)
    ax.axis("equal")
    st.pyplot(fig)


def render_trend_chart(df: pd.DataFrame, date_column: str):
    trend_df = df.copy()
    trend_df[date_column] = pd.to_datetime(trend_df[date_column], errors="coerce")
    trend_df = trend_df.dropna(subset=[date_column])

    daily_counts = (
        trend_df.groupby([trend_df[date_column].dt.date, "sentiment"])
        .size()
        .unstack(fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    for sentiment in ["Positive", "Neutral", "Negative"]:
        if sentiment in daily_counts.columns:
            ax.plot(daily_counts.index, daily_counts[sentiment],
                    marker="o", label=sentiment, color=SENTIMENT_COLORS[sentiment])
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Posts")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)


def render_wordcloud(df: pd.DataFrame, sentiment_label: str):
    text_blob = " ".join(
        df.loc[df["sentiment"] == sentiment_label, "cleaned_text"].astype(str)
    )
    if not text_blob.strip():
        st.info(f"No {sentiment_label.lower()} text to display.")
        return
    wc = WordCloud(width=500, height=300, background_color="white").generate(text_blob)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)


def main():
    st.title("📊 Social Media Sentiment Dashboard")
    st.caption("NLTK VADER-powered sentiment analysis for reviews, feedback, and social posts")

    with st.sidebar:
        st.header("Data Input")
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        use_sample = st.checkbox("Use sample dataset instead", value=uploaded_file is None)

        if uploaded_file is not None and not use_sample:
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = load_sample_data()

        text_column = st.selectbox(
            "Which column contains the text to analyze?",
            options=df_raw.columns,
            index=list(df_raw.columns).index("text") if "text" in df_raw.columns else 0,
        )

        date_columns = [c for c in df_raw.columns if "date" in c.lower()]
        date_column = st.selectbox(
            "Which column contains the date? (optional, for trend chart)",
            options=["None"] + date_columns,
            index=1 if date_columns else 0,
        )

    with st.spinner("Running sentiment analysis..."):
        df = analyze_dataframe(df_raw, text_column)

    counts = sentiment_summary(df)
    total = len(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Posts", total)
    col2.metric("😊 Positive", f"{counts.get('Positive', 0)} ({counts.get('Positive', 0)/total:.0%})")
    col3.metric("😐 Neutral", f"{counts.get('Neutral', 0)} ({counts.get('Neutral', 0)/total:.0%})")
    col4.metric("😠 Negative", f"{counts.get('Negative', 0)} ({counts.get('Negative', 0)/total:.0%})")

    st.divider()

    left, right = st.columns(2)
    with left:
        st.subheader("Sentiment Distribution")
        render_pie_chart(counts)
    with right:
        if date_column != "None":
            st.subheader("Sentiment Trend Over Time")
            render_trend_chart(df, date_column)
        else:
            st.info("Select a date column in the sidebar to see a trend chart.")

    st.divider()

    st.subheader("Word Clouds")
    wc_left, wc_right = st.columns(2)
    with wc_left:
        st.markdown("**Positive posts**")
        render_wordcloud(df, "Positive")
    with wc_right:
        st.markdown("**Negative posts**")
        render_wordcloud(df, "Negative")

    st.divider()

    st.subheader("Detailed Results")
    filter_choice = st.multiselect(
        "Filter by sentiment", options=["Positive", "Neutral", "Negative"],
        default=["Positive", "Neutral", "Negative"]
    )
    filtered_df = df[df["sentiment"].isin(filter_choice)]
    st.dataframe(
        filtered_df[[text_column, "sentiment", "compound"]],
        use_container_width=True,
        height=300,
    )

    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download results as CSV", data=csv_bytes,
        file_name="sentiment_results.csv", mime="text/csv"
    )

    st.divider()

    st.subheader("Try it yourself")
    user_text = st.text_area("Type a review, comment, or tweet:")
    if st.button("Analyze sentiment"):
        if user_text.strip():
            result = analyze_text(user_text)
            st.markdown(f"**Sentiment: :{'green' if result['sentiment']=='Positive' else 'red' if result['sentiment']=='Negative' else 'gray'}[{result['sentiment']}]** (compound score: {result['compound']:.3f})")
        else:
            st.warning("Please enter some text first.")


if __name__ == "__main__":
    main()
