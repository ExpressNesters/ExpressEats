from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("saved_final_xgboost.pkl", "rb"))

@app.route('/')
def home():
    return "Welcome to the Loan Approval Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract the input features from the received JSON object
        data = request.json
        # Ensure the features are in the correct order
        features = [data['loan_id'], data['no_of_dependents'], data['education'], data['self_employed'], 
                    data['income_annum'], data['loan_amount'], data['loan_term'], data['cibil_score'], 
                    data['residential_assets_value'], data['commercial_assets_value'], 
                    data['luxury_assets_value'], data['bank_asset_value']]

        # Convert features to DataFrame
        input_df = pd.DataFrame([features], columns=['loan_id', ' no_of_dependents', ' education', ' self_employed', 
                                                     ' income_annum', ' loan_amount', ' loan_term', ' cibil_score', 
                                                     ' residential_assets_value', ' commercial_assets_value', 
                                                     ' luxury_assets_value', ' bank_asset_value'])

        # Predict using the model
        prediction = model.predict(input_df)

        # Respond with the prediction
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8100)
