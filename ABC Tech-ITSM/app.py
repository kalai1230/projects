from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd

# Load the model
with open('model.pkl', 'rb') as model_file:
    log = pickle.load(model_file)

# Define the expected features (18 features)
expected_columns = [
    'CI_Name', 'CI_Cat', 'CI_Subcat', 'WBS', 'Incident_ID', 'Impact', 
    'Urgency', 'number_cnt', 'Category', 'KB_number', 
    'No_of_Reassignments', 'Handle_Time_hrs', 'Closure_Code', 
    'No_of_Related_Interactions', 'No_of_Related_Changes', 
    'Open_to_Resolved_Hours', 'Resolved_to_Close_Hours', 
    'Open_to_Close_Hours'
]

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from request
    data = request.form.to_dict()
    
    # Convert data to DataFrame
    df = pd.DataFrame([data])

    # Ensure columns match the model's training columns
    df = df.reindex(columns=expected_columns, fill_value=0)  # Ensure all columns are present with default values if missing

    # Predict
    prediction = log.predict(df)

    # Return prediction as JSON
    return jsonify({'Prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
