from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the model, vectorizer, and dataset
knn_retriever = joblib.load('ee_knn_retriever_model.pkl')
tfidf_vectorizer = joblib.load('ee_tfidf_vectorizer.pkl')
dataset = pd.read_csv('recipes_dataset.csv')
max_time = dataset['TotalTimeInMins'].max()

def make_recommendations(input_ingredients, input_time, model, tfidf_vectorizer, data, num_recommendations=5):
    # Process the input
    input_vector = tfidf_vectorizer.transform([input_ingredients])
    
    # Convert and reshape normalized time
    normalized_time = np.array([input_time / max_time]).reshape(-1, 1)
    input_features = np.hstack([input_vector.toarray(), normalized_time])

    # Find nearest recipes
    distances, indices = model.kneighbors(input_features, n_neighbors=num_recommendations)

    # Prepare recommendations
    recommendations = [{
        'name': data.iloc[index]['TranslatedRecipeName'],
        'description': data.iloc[index]['TranslatedInstructions']
    } for index in indices[0]]

    return recommendations

@app.route('/recommend', methods=['POST'])
def recommend():
    # Extract data from request
    data = request.json
    input_ingredients = data['ingredients']
    input_time = data.get('prep_time', 60)  # Default prep time

    # Get recommendations
    recommendations = make_recommendations(input_ingredients, input_time, knn_retriever, tfidf_vectorizer, dataset)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8100)
