# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model build by Ahmed Alaa, Using Gradient Boosting Classifier with HyperParameter set :`{'learning_rate': 0.1,
 'max_depth': 5,
 'min_samples_leaf': 1,
 'min_samples_split': 3,
 'n_estimators': 100}`

## Intended Use
This model intended to estimate the salary of an person using given financial info
## Training Data
- This data was extracted from the census bureau database found at
| http://www.census.gov/ftp/pub/DES/www/welcome.html
- [datalink](https://archive.ics.uci.edu/ml/datasets/census+income) Used 0.8 of the whole dataset as training_set

## Evaluation Data
- This data was extracted from the census bureau database found at
| http://www.census.gov/ftp/pub/DES/www/welcome.html
- [datalink](https://archive.ics.uci.edu/ml/datasets/census+income) Used 0.2 of the whole dataset as training_set

## Metrics
| #fbeta  | #precision  | #recall   |
| :---:   | :---:       | :---:     |
| 0.64644 | 0.71201     | 0.59193   |

## Ethical Considerations
This data contains gender, race and country, which might cause the model to bias toward one of these aspects, and that can potentially cause discrimination against persons.
## Caveats and Recommendations
This model is not very well generalized, and only can be used for statistical reports, not for decision making
