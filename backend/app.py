import pandas as pd
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and vectorizer
with open('recipe_model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Load the cleaned dataset
cleaned_dataset = pd.read_csv('recipes_dataset.csv')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    ingredients = data['ingredients']
    prep_time = data.get('prep_time', 60)  # Default prep time, e.g., 30 minutes

    # Preprocess and vectorize the input
    vectorized_ingredients = vectorizer.transform([ingredients])
    max_prep_time = 510  # Update this with the max prep time in your dataset
    normalized_prep_time = prep_time / max_prep_time
    combined_features = np.hstack((vectorized_ingredients.toarray(), [[normalized_prep_time]]))

    # Make a prediction
    distances, indices = model.kneighbors(combined_features)
    recommendations = [{
        'name': cleaned_dataset.iloc[index]['TranslatedRecipeName'],
        'instructions': cleaned_dataset.iloc[index]['TranslatedInstructions']  # Or 'Instructions'
    } for index in indices[0]]

    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8100)
