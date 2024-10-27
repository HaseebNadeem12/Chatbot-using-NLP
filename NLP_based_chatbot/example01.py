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

# Load the JSON data
my_file = pd.read_json("C:/Users/COMTECH COMPUTER/Desktop/Datasets for Chatbot/intents.json")


#-> Checking that data (stopwords, tokenizer) is downloaded properly or not
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize Porter Stemmer and WordNet Lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    # 1. Convert all text to lowercase
    text = text.lower()

    # 3. Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)

    # 4. Tokenize text
    word_tokens = word_tokenize(text)

    # 5. Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in word_tokens if word not in stop_words]

    # 6. Apply stemming and lemmatization
    stemmed_words = [stemmer.stem(word) for word in words]    # Stemming is not giving us correct word
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

    # Join back into a single string
    return " ".join(lemmatized_words)


list_of_intents  = []
list_of_question = []
list_of_answer   = []

for i in range(len(my_file)):

    for j in range(len(my_file['intents'][i]['patterns'])):

        list_of_intents.append(preprocess_text(my_file['intents'][i]['tag']))  # creating intent for each question

        processed_text02 = preprocess_text(my_file['intents'][i]['patterns'][j])
        list_of_question.append(processed_text02)

    for k in range(len(my_file['intents'][i]['responses'])):
        processed_text03 = preprocess_text(my_file['intents'][i]['responses'][k])
        list_of_answer.append(processed_text03)

# print(len(list_of_intents),  list_of_intents)
# print(len(list_of_question), list_of_question)
# print(len(list_of_answer),   list_of_answer)

TF_IDF = TfidfVectorizer(max_features=1000)
X_features = TF_IDF.fit_transform(list_of_question).toarray()
y_features = list_of_intents

# X_train,X_test,y_train,y_test = train_test_split(X_features,y_features,
#                                                  test_size=0.2,train_size=0.8, random_state=42)
#
# svm_model = SVC(kernel='linear', C=10)      # Achiveing high acuracy with linear and C=10
# svm_model.fit(X_train,y_train)
# predictions = svm_model.predict(X_test)
# print(X_test)

#-> Training on complete data
SVM_model = SVC(kernel='linear', C=10)
SVM_model.fit(X_features, y_features)


# #-> To find which combination work best
# parameters = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
# grid_search = GridSearchCV(SVC(), parameters)
# grid_search.fit(X_train, y_train)
#
# print(grid_search.best_params_)  # Find best parameters
#
# # -> Predictions
# print(y_test,'\n')
# print(predictions)

#-> Checking the model's overall accuracy
# accuracy = accuracy_score(y_test, predictions)
# print(f"Model Accuracy: {accuracy * 100:.2f}%",'\n')
#
# print(classification_report(y_test, predictions))
"""============================================================================"""

"""Applying sentiment analysis"""

# Function to classify sentiment based on compound score
def classify_sentiment(score):
    if score['compound'] >= 0.05:
        return 1  # positive sentiment as 1
    elif score['compound'] <= -0.05:
        return -1  # negative sentiment as -1
    else:
        return 0  # neutral sentiment as 0


def sentiment_classifier(text):

    sia = SentimentIntensityAnalyzer()

    score = sia.polarity_scores(text)
    result = classify_sentiment(score)
    return result


def detect_intent(input_text, X_features):
    input_vector = TF_IDF.transform([input_text]).toarray()
    confidence_score = cosine_similarity(input_vector, X_features).max()
    return confidence_score


def custom_state_machine(intent, state, sentiment_score, similarity_score):

    if state == "greeting":
        print("Hi , I am your paramedic assistant")
        return "waiting_for_input"

    elif state == "waiting_for_input" and similarity_score >= 0.5:

        if sentiment_score == -1:
            print("Don't be panic")
            print("Just follow the instruction ")

        for i in range(len(list_of_intents)):

            if intent == preprocess_text(my_file['intents'][i]['tag']):
                print(my_file['intents'][i]['responses'][0])

                print("\nWould you like more information on something else?")
                user_input = input("Enter Yes or No: ")

                if user_input.lower() == "yes":
                    return "waiting_for_input"

                elif user_input.lower() == 'no':
                    print("Thank you for using the chatbot. Goodbye!")
                    return "end"

                else:
                    print("You did not entered the correct input")
                    print("I am considering that, You are trying to ask a Question")
                    return "waiting_for_input"


    elif similarity_score < 0.5:
        print("I did not understand you question, can you please repeat your question")
        print("I can help with any medical emergency situation")
        return "waiting_for_input"


current_state = 'greeting'
# Looping through states based on user inputs
while current_state != "end":

    if current_state == "greeting":
        current_state = custom_state_machine(None, current_state, None, None)

    elif current_state !="greeting" :
        user_input = input("Please type your question1: ")
        processed_input = preprocess_text(user_input)
        intent = SVM_model.predict(TF_IDF.transform([processed_input]).toarray())[0]

        similarity_score = detect_intent(user_input,X_features)
        sentiment_result = sentiment_classifier(user_input)

        current_state = custom_state_machine(intent, current_state,sentiment_result,similarity_score)

