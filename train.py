import joblib

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

filepath = 'data/clean_toxic_comment_dataset.csv'
df_train = pd.read_csv(filepath)

#  model
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
text_features = 'cleaned_comment_text'

X = df_train[[text_features]]
y = df_train[labels]

preprocessor = ColumnTransformer(
    transformers=[
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=75000), text_features)
    ]
)

sgd_svm_base = SGDClassifier(
    loss='hinge',
    penalty='l2',
    alpha=1e-4,
    max_iter=1000,
    class_weight='balanced',
    early_stopping=True,
    random_state=42,
    n_jobs=-1
)

calibrated_svm = CalibratedClassifierCV(
    estimator=sgd_svm_base,
    method='sigmoid',
    cv=3
)

ovr_svm = OneVsRestClassifier(calibrated_svm)

final_model = Pipeline([
    ('prep', preprocessor),
    ('clf', ovr_svm)
])

final_model.fit(X, y)

# to save model
joblib.dump(final_model, 'toxic_comment_prediction_model.pkl')
print("Saved toxic_comment_prediction_model.pkl")