import pandas as pd

# Path should be according to youe system
df = pd.read_csv("C:/Users/vanag/OneDrive/Desktop/vedh school stuff/archive/all-data.csv", encoding='latin-1',names=['sentiment','News Heading'])
print(df.head())
print(df.shape)
df.dropna(inplace=True)
print(df.shape)
print(df.head())

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Split the data into training and testing sets
X = df['News Heading']  # Input features (news headings)
y = df['sentiment']     # Target variable (sentiment labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the text data into numerical features using TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Train the model (Random Forest classifier)
model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train_vectors, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_vectors)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", report)

# Single news heading for prediction
single_news_heading = "SpaceX scrubs Falcon Heavy launch Wednesday from Kennedy Space Center - Florida Today"

# Convert the text into numerical features using TF-IDF Vectorizer
numerical_vector = vectorizer.transform([single_news_heading])

# Make the prediction using the trained Random Forest classifier
predicted_sentiment = model.predict(numerical_vector)

print(f"Predicted Sentiment: {predicted_sentiment[0]}")

from joblib import dump

model_file_path = "random_forest_model.joblib"

# Save the model to the specified file path
dump(model, model_file_path)

print(f"Model saved to {model_file_path}")

vectorizer_file_path = "tfidf_vectorizer.joblib"

dump(vectorizer, vectorizer_file_path)

print(f"Vectorizer saved to {vectorizer_file_path}")
