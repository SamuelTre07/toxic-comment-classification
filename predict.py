from cleaning import clean_text_pipeline

import joblib
import json

import pandas as pd

from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

# pydantic validation
# request
class TextInput(BaseModel):
    text: str

class PredOutput(BaseModel):
    toxicity_probability: dict

# app instantiation
app = FastAPI(title='toxic-comment-prediction')

# loading the model
pipeline = joblib.load('toxic_comment_prediction_model.pkl')

# loading the thresholds
with open('thresholds.json') as f:
    thresholds = json.load(f)

# labels to be classified
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# function to perform the prediction
def predict_comment(comment):
    clean_comment = clean_text_pipeline(comment)
    clean_comment = pd.DataFrame({'cleaned_comment_text': [clean_comment]})
    y_pred_proba = pipeline.predict_proba(clean_comment)[0]
    y_pred_proba = [float(pred) for pred in y_pred_proba]
    # applying thresholds
    results = {}
    for label, prob in zip(labels, y_pred_proba):
        results[label] = {'score': prob,
                          'is_toxic': prob >= thresholds[label]}
    return results

# exposing api endpoint
@app.post('/predict')
def predict(input: TextInput) -> PredOutput:
    prediction = predict_comment(input.text)
    
    return PredOutput(
        toxicity_probability=prediction
    )

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)