from flask import Flask, request, jsonify
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data is downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)

# Load the TF-IDF vectorizer and label encoder
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
label_encoder = LabelEncoder()

# Load your datasets
df = pd.read_csv('Symptom2Disease.csv')
df_second = pd.read_csv('symptom_precaution.csv')

# Preprocess the datasets
X = tfidf_vectorizer.fit_transform(df['text'].astype(str))
y = label_encoder.fit_transform(df['label'])

# Data cleaning and preprocessing for the second dataset
df_second['Disease'] = df_second['Disease'].str.lower().str.strip().apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
for col in ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']:
    df_second[col] = df_second[col].str.lower().str.strip().apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)) if isinstance(x, str) else '')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if isinstance(text, str):
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return tokens
    else:
        return []

for col in ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']:
    df_second[f'{col}_tokens'] = df_second[col].apply(preprocess_text)

df_second.fillna('', inplace=True)

# Store user state
user_state = {}

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('user_input')
    user_id = data.get('user_id')  # Assuming user_id is provided to identify the user

    def predict_disease(user_input):
        user_input_vector = tfidf_vectorizer.transform([user_input])
        similarity_scores = cosine_similarity(user_input_vector, X)
        matched_index = np.argmax(similarity_scores)
        if similarity_scores[0][matched_index] > 0.4:
            matched_disease = label_encoder.inverse_transform([y[matched_index]])[0]
            return matched_disease
        else:
            return None

    def get_actions(matched_disease):
        matched_disease = matched_disease.lower().strip()
        matched_row = df_second[df_second['Disease'] == matched_disease]

        if not matched_row.empty:
            precautions = {
                'precaution_1': matched_row['Precaution_1'].values[0] if matched_row['Precaution_1'].values else 'No precaution 1 found',
                'precaution_2': matched_row['Precaution_2'].values[0] if matched_row['Precaution_2'].values else 'No precaution 2 found',
                'precaution_3': matched_row['Precaution_3'].values[0] if matched_row['Precaution_3'].values else 'No precaution 3 found',
                'precaution_4': matched_row['Precaution_4'].values[0] if matched_row['Precaution_4'].values else 'No precaution 4 found'
            }
            return precautions
        else:
            return {'error': 'No actions or precautions found for this disease.'}

    def chatbot_response(user_input, user_id):
        user_input = user_input.lower().strip()
        if user_input in ["hi", "hello"]:
            return "Hello! How can I assist you today?"
        elif user_input in ["bye", "goodbye"]:
            return "Goodbye! Have a great day!"
        elif user_input == "how are you":
            return "I'm just a computer program, so I don't have feelings, but I'm here to assist you!"
        elif user_input in ["thank you", "thanks"]:
            return "You're welcome! If you have any more questions, feel free to ask."
        elif user_input == "yes" and user_id in user_state:
            user_data = user_state[user_id]
            if user_data['action_provided']:
                return "You've already received the actions/precautions. If you have any other questions, feel free to ask."
            else:
                # User wants to know actions for the matched disease
                matched_disease = user_data['matched_disease']
                actions = get_actions(matched_disease)
                user_data['action_provided'] = True  # Mark actions as provided
                if 'error' in actions:
                    return actions['error']
                else:
                    return (
                        f"Actions/Precautions for {matched_disease}:\n"
                        f"1. {actions['precaution_1']}\n"
                        f"2. {actions['precaution_2']}\n"
                        f"3. {actions['precaution_3']}\n"
                        f"4. {actions['precaution_4']}"
                    )
        elif user_input == "no" and user_id in user_state:
            return "Chatbot: Okay, if you have any other questions, feel free to ask."
        else:
            matched_disease = predict_disease(user_input)
            if matched_disease:
                actions_available = matched_disease in df_second['Disease'].values
                user_state[user_id] = {'matched_disease': matched_disease, 'action_provided': False}  # Save the matched disease in the user state
                if actions_available:
                    return f"The matched disease for your description is: {matched_disease}. Do you want to know what actions/precautions you should take? (yes/no)"
                else:
                    return f"The matched disease for your description is: {matched_disease}. However, no actions/precautions are available for this disease. Do you want to know what actions/precautions you should take? (yes/no)"
            else:
                return "Sorry, the disease corresponding to your description was not found in the dataset."

    if user_input:
        # General chatbot response, including disease prediction and actions/precautions
        response = chatbot_response(user_input, user_id)
        if isinstance(response, dict):
            return jsonify(response)
        else:
            return jsonify(response)
    else:
        return jsonify({'error': 'Invalid input'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
