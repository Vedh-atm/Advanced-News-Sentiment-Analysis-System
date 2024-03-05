# Set up the Streamlit app
import streamlit as st
import pandas as pd
from gnewsclient import gnewsclient
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os


# Load pre-trained models and vectorizer
loaded_vectorizer = joblib.load("tfidf_vectorizer.joblib")
loaded_model = joblib.load("random_forest_model.joblib")
vectorizer = loaded_vectorizer
model = loaded_model

# Initialize news_df as a global variable
news_df = None

# Function to perform sentiment analysis on a given headline
def analyze_sentiment(news_df, vectorizer, model):
    # Transform the titles into vectors
    title_vectors = vectorizer.transform(news_df['Title'])

    # Predict sentiment for title vectors
    sentiments = model.predict(title_vectors)

    # Add the sentiment column to the DataFrame
    news_df['Sentiment'] = sentiments

    # Count occurrences of each sentiment category
    sentiment_counts = news_df['Sentiment'].value_counts()

    return sentiment_counts


# Function to fetch news headlines for a given topic and save to Excel
def fetch_and_save_news(topic):
    global news_df
    location = 'India'
    max_results = 10

    # Create an empty list to store news data
    news_data = []

    for topic in [topic]:
        client = gnewsclient.NewsClient(language='english', location=location, topic=topic, max_results=max_results)
        news_list = client.get_news()
        for item in news_list:
            # Append news data to the list
            news_data.append({
                'Topic': topic,
                'Title': item['title'],
                'Link': item['link']
            })

    # Convert the list of dictionaries to a DataFrame
    new_news_df = pd.DataFrame(news_data)

    # Save news to Excel
    excel_file_name = f'C:/Users/vanag/Downloads/news_data_{topic.lower().replace(" ", "_")}.xlsx'
    
    if os.path.exists(excel_file_name):
        # If the file already exists, read the existing data and append the new data
        existing_news_df = pd.read_excel(excel_file_name)
        news_df = pd.concat([existing_news_df, new_news_df], ignore_index=True)
        news_df.to_excel(excel_file_name, index=False)
        st.write(f"News data appended to {excel_file_name}")
    else:
        # If the file doesn't exist, save the new data to a new file
        new_news_df.to_excel(excel_file_name, index=False)
        st.write(f"News data saved to {excel_file_name}")
# Set up the Streamlit app
st.title("News Sentiment Analysis System")


# Topic Selection
topic = st.selectbox("Select a topic:", ['Health', 'Science', 'Technology', 'Politics'])

# Fetch and Save News to Excel
if st.button("Fetch and Save News to Excel"):
    fetch_and_save_news(topic)

# Analyze Sentiment
if st.button("Analyze Sentiment"):
    excel_file_name = f'C:/Users/vanag/Downloads/news_data_{topic.lower().replace(" ", "_")}.xlsx'
    try:
        news_df = pd.read_excel(excel_file_name)

        if news_df is None:
            st.write("No news data available for sentiment analysis.")
        elif news_df.empty:
            st.write("The news data is empty. Please fetch news data first.")
        else:
            sentiment_counts = analyze_sentiment(news_df, vectorizer, model)
            st.write("Sentiment Analysis Result:")
            st.write(f"Positive News: {sentiment_counts.get('positive', 0)}")
            st.write(f"Negative News: {sentiment_counts.get('negative', 0)}")
            st.write(f"Neutral News: {sentiment_counts.get('neutral', 0)}")
    except FileNotFoundError:
        st.write(f"No news data available for sentiment analysis. Please fetch and save news data first for topic: {topic}.")

# Show Topic, Title, Link, and Sentiment
if news_df is not None and not news_df.empty:
    st.header("News Data")
    for index, row in news_df.iterrows():
        st.subheader(f"Topic: {row['Topic']}")
        st.write(f"Title: {row['Title']}")
        st.write(f"Link: {row['Link']}")
        
        # Perform sentiment analysis
        title_vector = vectorizer.transform([row['Title']])
        sentiment = model.predict(title_vector)[0]
        
        # Convert sentiment code to sentiment label
        if sentiment == 'positive':
            sentiment_label = 'Positive'
        elif sentiment == 'negative':
            sentiment_label = 'Negative'
        else:
            sentiment_label = 'Neutral'
            
        st.write(f"Sentiment: {sentiment_label}")
        st.write("---")

