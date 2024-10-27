import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Data Preprocessing Libraries
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.pairwise import cosine_similarity

# Load JSON data containing intents and patterns for the chatbot
my_file = pd.read_json("C:/Users/COMTECH COMPUTER/Desktop/Datasets for Chatbot/intents.json")

# Check that essential NLTK datasets are available (stopwords, tokenizer, lemmatizer)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize stemming and lemmatization tools
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Convert text to lowercase to maintain consistency
    text = text.lower()

    # Remove punctuation and special characters to clean the text
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize text to break it into individual words
    word_tokens = word_tokenize(text)

    # Filter out stopwords to retain only meaningful words
    stop_words = set(stopwords.words('english'))
    words = [word for word in word_tokens if word not in stop_words]

    # Apply lemmatization to get the root form of each word (better for semantic processing)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

    # Combine the processed words back into a single string
    return " ".join(lemmatized_words)

# Initializing lists to store intents, questions, and answers
list_of_intents = []
list_of_question = []
list_of_answer = []

# Iterate over the JSON data to populate intent, question, and answer lists
for i in range(len(my_file)):

    # Process each pattern (question) and corresponding intent
    for j in range(len(my_file['intents'][i]['patterns'])):

        # Store processed intent tag and question pattern
        list_of_intents.append(preprocess_text(my_file['intents'][i]['tag']))
        processed_text02 = preprocess_text(my_file['intents'][i]['patterns'][j])
        list_of_question.append(processed_text02)

    # Process each response related to an intent and store it
    for k in range(len(my_file['intents'][i]['responses'])):

        processed_text03 = preprocess_text(my_file['intents'][i]['responses'][k])
        list_of_answer.append(processed_text03)

# Apply TF-IDF to vectorize questions (convert text to numerical representation)
TF_IDF = TfidfVectorizer(max_features=1000)
X_features = TF_IDF.fit_transform(list_of_question).toarray()  # Features (questions)
y_features = list_of_intents  # Labels (intents)

# Initialize and train the Support Vector Machine (SVM) model on all available data
SVM_model = SVC(kernel='linear', C=10)
SVM_model.fit(X_features, y_features)  # Train SVM on question features and intents

# Function to classify sentiment based on compound score (positive, negative, neutral)
def classify_sentiment(score):

    if score['compound'] >= 0.05:
        return 1  # Positive sentiment
    elif score['compound'] <= -0.05:
        return -1  # Negative sentiment
    else:
        return 0  # Neutral sentiment

# Sentiment classifier function that processes user input to determine sentiment
def sentiment_classifier(text):

    sia = SentimentIntensityAnalyzer()  # Initialize sentiment intensity analyzer
    score = sia.polarity_scores(text)  # Get sentiment scores for input text
    result = classify_sentiment(score)  # Classify sentiment based on compound score
    return result

# Function to detect intent confidence by measuring similarity to trained data
def detect_intent(input_text, X_features):

    input_vector = TF_IDF.transform([input_text]).toarray()  # Vectorize user input
    confidence_score = cosine_similarity(input_vector, X_features).max()  # Max similarity score
    return confidence_score

# State machine to manage dialogue flow and responses based on user input, intent, and sentiment
def custom_state_machine(intent, state, sentiment_score, similarity_score):

    if state == "greeting":

        print("Hi, I am your paramedic assistant")
        return "waiting_for_input"  # Transition to next state to accept input

    elif state == "waiting_for_input" and similarity_score >= 0.5:

        # Check if the sentiment is negative; provide appropriate response if so
        if sentiment_score == -1:

            print("Don't be panic")
            print("Just follow the instruction")

        # Respond to recognized intent and provide option for further questions
        for i in range(len(list_of_intents)):

            if intent == preprocess_text(my_file['intents'][i]['tag']):

                print('\n',my_file['intents'][i]['responses'][0])  # Provide response

                print("\nWould you like more information on something else?")
                user_input = input("Enter Yes or No: ")

                # Determine next state based on user’s response
                if user_input.lower() == "yes":

                    return "waiting_for_input"

                elif user_input.lower() == 'no':

                    print("Thank you for using the chatbot. Goodbye!")
                    return "end"

                else:
                    # Fallback if input is unclear
                    print("\nYou did not enter the correct input")
                    print("I am considering that, You are trying to ask a Question\n")
                    return "waiting_for_input"

    elif similarity_score < 0.5:

        # Handle cases with low similarity score; prompt user for clarification
        print("\nI did not understand your question, can you please repeat?")
        print("I can help with any medical emergency situation\n")
        return "waiting_for_input"

# Initialize dialogue state and enter conversation loop
current_state = 'greeting'

while current_state != "end":

    if current_state == "greeting":
        # Initial greeting state
        current_state = custom_state_machine(None, current_state, None, None)

    else:
        # Capture user input and determine intent, sentiment, and similarity
        user_input = input("Please type your question: ")
        processed_input = preprocess_text(user_input)
        intent = SVM_model.predict(TF_IDF.transform([processed_input]).toarray())[0]

        similarity_score = detect_intent(user_input, X_features)
        sentiment_result = sentiment_classifier(user_input)

        # Determine next state based on detected intent, sentiment, and similarity score
        current_state = custom_state_machine(intent, current_state, sentiment_result, similarity_score)
