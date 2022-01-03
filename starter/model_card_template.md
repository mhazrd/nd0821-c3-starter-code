# Model Card

This model card describes about the income prediction model trained on census income dataset.

## Model Details

The Scikit-Learn's Random Forest model is trained on the census income dataset.

## Intended Use

This model is created to prove that it's possible to predict a person's income level (>50k, <=50k) based on just a few characteristics of the person including demographics, occupation information.

## Training Data

Census Income Dataset (https://archive.ics.uci.edu/ml/datasets/census+income)

The dataset contains 48842 salary-level instances with an associated person's 14 attributes. 80% are randomly chosen to train a model

## Evaluation Data

Census Income Dataset (https://archive.ics.uci.edu/ml/datasets/census+income)

The dataset contains 48842 salary-level instances with an associated person's 14 attributes. 20% are randomly chosen to test the trained model's performance.

## Metrics

The model is evaluated on precision, recall and f1 scores. (rounded to 3 decimal places)

- Precision: 0.722
- Recall: 0.614
- F1 score: 0.664

## Ethical Considerations

The model is trained on a public dataset from UCI ML repository. The dataset is gathered in 1994 which can make the model not reflect the current status of world

## Caveats and Recommendations

The model is just created for a demonstration purpose so it's not highly optimised for performance. It's not recommended to use the model for an actual production use-case to estimate a person's income-level