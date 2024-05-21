# from flask import Flask, request, jsonify
# import numpy as np
# from tensorflow.keras.models import load_model
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics.pairwise import cosine_similarity
# import pandas as pd

# # Load your trained model
# model = load_model('model.h5')

# # Load the TF-IDF vectorizer and label encoder
# tfidf_vectorizer = TfidfVectorizer(max_features=1000)
# label_encoder = LabelEncoder()

# # Load your dataset or any necessary data
# df = pd.read_csv('Symptom2Disease.csv')

# # Preprocess the data
# X = tfidf_vectorizer.fit_transform(df['text'].astype(str))
# y = label_encoder.fit_transform(df['label'])

# app = Flask(__name__)

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get input text from request
#     data = request.get_json()
#     user_input = data['user_input']

#     # Preprocess the input text
#     user_input_vector = tfidf_vectorizer.transform([user_input])

#     # Use cosine similarity to find the most similar disease
#     similarity_scores = cosine_similarity(user_input_vector, X)
#     matched_index = np.argmax(similarity_scores)

#     # Get the predicted disease
#     matched_disease = label_encoder.inverse_transform([y[matched_index]])[0]

#     return jsonify({'prediction': matched_disease})

# @app.route('/actions', methods=['POST'])
# def actions():
#     # Get input from request
#     data = request.get_json()
#     user_response = data['user_response']
#     matched_disease = data['matched_disease']

#     # Retrieve actions/precautions for the matched disease
#     matched_row = df[df['label'] == matched_disease]

#     if not matched_row.empty:
#         precautions = {
#             'precaution_1': matched_row['Precaution_1'].values[0],
#             'precaution_2': matched_row['Precaution_2'].values[0],
#             'precaution_3': matched_row['Precaution_3'].values[0],
#             'precaution_4': matched_row['Precaution_4'].values[0]
#         }

#         return jsonify(precautions)
#     else:
#         return jsonify({'error': 'No actions or precautions found for this disease.'})
# @app.route('/chatbot', methods=['POST'])
# def chatbot():
#     # Get input from request
#     data = request.get_json()
#     user_input = data['user_input']

#     # Define chatbot response function
#     def chatbot_response(user_input):
#         user_input = user_input.lower().strip()

#         if user_input == "hi" or user_input == "hello":
#             return "Hello! How can I assist you today?"

#         elif user_input == "bye" or user_input == "goodbye":
#             return "Goodbye! Have a great day!"

#         elif user_input == "how are you":
#             return "I'm just a computer program, so I don't have feelings, but I'm here to assist you!"

#         elif user_input == "thank you" or user_input == "thanks":
#             return "You're welcome! If you have any more questions, feel free to ask."

#         else:
#             user_input_vector = tfidf_vectorizer.transform([user_input])
#             similarity_scores = cosine_similarity(user_input_vector, X)
#             matched_index = np.argmax(similarity_scores)

#             if similarity_scores[0][matched_index] > 0.2:
#                 matched_disease = y[matched_index]
#                 actions_available = matched_disease in df['Disease'].values

#                 if actions_available:
#                     return f"The matched disease for your description is: {matched_disease}. Do you want to know what actions/precautions you should take? (yes/no)"
#                 else:
#                     return f"The matched disease for your description is: {matched_disease}. However, no actions/precautions are available for this disease. Do you want to know what actions/precautions you should take? (yes/no)"
#             else:
#                 return "Sorry, the disease corresponding to your description was not found in the dataset."

#     # Get chatbot response
#     response = chatbot_response(user_input)

#     return jsonify({'response': response})

# if __name__ == '__main__':
#     app.run(debug=True)



# from flask import Flask, request, jsonify
# import numpy as np
# from tensorflow.keras.models import load_model
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics.pairwise import cosine_similarity
# import pandas as pd
# import string
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer

# # Ensure NLTK data is downloaded
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

# app = Flask(__name__)

# # Load your trained model
# model = load_model('model.h5')

# # Load the TF-IDF vectorizer and label encoder
# tfidf_vectorizer = TfidfVectorizer(max_features=1000)
# label_encoder = LabelEncoder()

# # Load your dataset or any necessary data
# df = pd.read_csv('Symptom2Disease.csv')

# # Preprocess the data
# X = tfidf_vectorizer.fit_transform(df['text'].astype(str))
# y = label_encoder.fit_transform(df['label'])

# df_second = pd.read_csv('symptom_precaution.csv')

# # Data cleaning and preprocessing
# df_second['Disease'] = df_second['Disease'].str.lower().str.strip()
# df_second['Precaution_1'] = df_second['Precaution_1'].str.lower().str.strip()
# df_second['Precaution_2'] = df_second['Precaution_2'].str.lower().str.strip()
# df_second['Precaution_3'] = df_second['Precaution_3'].str.lower().str.strip()
# df_second['Precaution_4'] = df_second['Precaution_4'].str.lower().str.strip()

# df_second['Disease'] = df_second['Disease'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
# df_second['Precaution_1'] = df_second['Precaution_1'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
# df_second['Precaution_2'] = df_second['Precaution_2'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
# df_second['Precaution_3'] = df_second['Precaution_3'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)) if isinstance(x, str) else x)
# df_second['Precaution_4'] = df_second['Precaution_4'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)) if isinstance(x, str) else x)

# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()

# def preprocess_text(text):
#     if isinstance(text, str):
#         tokens = word_tokenize(text)
#         tokens = [word for word in tokens if word not in stop_words]
#         tokens = [lemmatizer.lemmatize(word) for word in tokens]
#         return tokens
#     else:
#         return []

# df_second['Precaution_1_tokens'] = df_second['Precaution_1'].apply(preprocess_text)
# df_second['Precaution_2_tokens'] = df_second['Precaution_2'].apply(preprocess_text)
# df_second['Precaution_3_tokens'] = df_second['Precaution_3'].apply(preprocess_text)
# df_second['Precaution_4_tokens'] = df_second['Precaution_4'].apply(preprocess_text)

# df_second.fillna('', inplace=True)

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     user_input = data['user_input']
#     user_input_vector = tfidf_vectorizer.transform([user_input])
#     similarity_scores = cosine_similarity(user_input_vector, X)
#     matched_index = np.argmax(similarity_scores)
#     matched_disease = label_encoder.inverse_transform([y[matched_index]])[0]
#     return jsonify({'prediction': matched_disease})

# @app.route('/actions', methods=['POST'])
# def actions():
#     data = request.get_json()
#     user_response = data['user_response']
#     matched_disease = data['matched_disease'].lower().strip()
#     matched_row = df_second[df_second['Disease'] == matched_disease]

#     if not matched_row.empty:
#         precautions = {
#             'precaution_1': matched_row['Precaution_1'].values[0] if matched_row['Precaution_1'].values else 'No precaution 1 found',
#             'precaution_2': matched_row['Precaution_2'].values[0] if matched_row['Precaution_2'].values else 'No precaution 2 found',
#             'precaution_3': matched_row['Precaution_3'].values[0] if matched_row['Precaution_3'].values else 'No precaution 3 found',
#             'precaution_4': matched_row['Precaution_4'].values[0] if matched_row['Precaution_4'].values else 'No precaution 4 found'
#         }
#         return jsonify(precautions)
#     else:
#         return jsonify({'error': 'No actions or precautions found for this disease.'})

# @app.route('/chatbot', methods=['POST'])
# def chatbot():
#     data = request.get_json()
#     user_input = data['user_input']

#     def chatbot_response(user_input):
#         user_input = user_input.lower().strip()
#         if user_input in ["hi", "hello"]:
#             return "Hello! How can I assist you today?"
#         elif user_input in ["bye", "goodbye"]:
#             return "Goodbye! Have a great day!"
#         elif user_input == "how are you":
#             return "I'm just a computer program, so I don't have feelings, but I'm here to assist you!"
#         elif user_input in ["thank you", "thanks"]:
#             return "You're welcome! If you have any more questions, feel free to ask."
#         else:
#             user_input_vector = tfidf_vectorizer.transform([user_input])
#             similarity_scores = cosine_similarity(user_input_vector, X)
#             matched_index = np.argmax(similarity_scores)
#             if similarity_scores[0][matched_index] > 0.2:
#                 matched_disease = label_encoder.inverse_transform([y[matched_index]])[0]
#                 actions_available = matched_disease in df_second['Disease'].values
#                 if actions_available:
#                     return f"The matched disease for your description is: {matched_disease}. Do you want to know what actions/precautions you should take? (yes/no)"
#                 else:
#                     return f"The matched disease for your description is: {matched_disease}. However, no actions/precautions are available for this disease. Do you want to know what actions/precautions you should take? (yes/no)"
#             else:
#                 return "Sorry, the disease corresponding to your description was not found in the dataset."

#     response = chatbot_response(user_input)
#     return jsonify({'response': response})

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=80)


from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
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

# Load your trained model
model = load_model('model.h5')

# Load the TF-IDF vectorizer and label encoder
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
label_encoder = LabelEncoder()

# Load your dataset or any necessary data
df = pd.read_csv('Symptom2Disease.csv')

# Preprocess the data
X = tfidf_vectorizer.fit_transform(df['text'].astype(str))
y = label_encoder.fit_transform(df['label'])

df_second = pd.read_csv('symptom_precaution.csv')

# Data cleaning and preprocessing
df_second['Disease'] = df_second['Disease'].str.lower().str.strip()
df_second['Precaution_1'] = df_second['Precaution_1'].str.lower().str.strip()
df_second['Precaution_2'] = df_second['Precaution_2'].str.lower().str.strip()
df_second['Precaution_3'] = df_second['Precaution_3'].str.lower().str.strip()
df_second['Precaution_4'] = df_second['Precaution_4'].str.lower().str.strip()

df_second['Disease'] = df_second['Disease'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
df_second['Precaution_1'] = df_second['Precaution_1'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
df_second['Precaution_2'] = df_second['Precaution_2'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
df_second['Precaution_3'] = df_second['Precaution_3'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)) if isinstance(x, str) else x)
df_second['Precaution_4'] = df_second['Precaution_4'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)) if isinstance(x, str) else x)

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

df_second['Precaution_1_tokens'] = df_second['Precaution_1'].apply(preprocess_text)
df_second['Precaution_2_tokens'] = df_second['Precaution_2'].apply(preprocess_text)
df_second['Precaution_3_tokens'] = df_second['Precaution_3'].apply(preprocess_text)
df_second['Precaution_4_tokens'] = df_second['Precaution_4'].apply(preprocess_text)

df_second.fillna('', inplace=True)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('user_input')
    user_response = data.get('user_response')
    matched_disease = data.get('matched_disease')

    def predict_disease(user_input):
        user_input_vector = tfidf_vectorizer.transform([user_input])
        similarity_scores = cosine_similarity(user_input_vector, X)
        matched_index = np.argmax(similarity_scores)
        matched_disease = label_encoder.inverse_transform([y[matched_index]])[0]
        return matched_disease

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

    def chatbot_response(user_input):
        user_input = user_input.lower().strip()
        if user_input in ["hi", "hello"]:
            return "Hello! How can I assist you today?"
        elif user_input in ["bye", "goodbye"]:
            return "Goodbye! Have a great day!"
        elif user_input == "how are you":
            return "I'm just a computer program, so I don't have feelings, but I'm here to assist you!"
        elif user_input in ["thank you", "thanks"]:
            return "You're welcome! If you have any more questions, feel free to ask."
        else:
            matched_disease = predict_disease(user_input)
            actions_available = matched_disease in df_second['Disease'].values
            if actions_available:
                return f"The matched disease for your description is: {matched_disease}. Do you want to know what actions/precautions you should take? (yes/no)"
            else:
                return f"The matched disease for your description is: {matched_disease}. However, no actions/precautions are available for this disease. Do you want to know what actions/precautions you should take? (yes/no)"

    if user_response and matched_disease:
        # User wants to know actions for the matched disease
        actions = get_actions(matched_disease)
        return jsonify(actions)
    elif user_input:
        # General chatbot response, including disease prediction
        response = chatbot_response(user_input)
        return jsonify({'response': response})
    else:
        return jsonify({'error': 'Invalid input'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
