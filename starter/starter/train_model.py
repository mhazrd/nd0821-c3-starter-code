# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import pickle

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

# Add code to load in the data.
data = pd.read_csv('../data/cleaned_census.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)
with open('../model/model.pkl', 'w+b') as f:
    pickle.dump(model, f)
with open('../model/encoder.pkl', 'w+b') as f:
    pickle.dump(encoder, f)


# Evaulate the model
y_pred = inference(model, X_test)
overall_metric = compute_model_metrics(y_test, y_pred)

slices = test['education'].unique()
slice_metrics = [('overall',)+overall_metric]
for slc in slices:
    filt = test['education'] == slc
    slice_metrics.append(
        (slc,) + compute_model_metrics(y_test[filt], y_pred[filt]))

metric_df = pd.DataFrame(slice_metrics, columns=['education_slice', 'precision', 'recall', 'f1'])
metric_df.to_csv('../model/slice_output.txt', index=False)