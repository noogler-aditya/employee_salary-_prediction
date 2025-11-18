# Employee Salary Prediction

A machine learning project that predicts whether an employee's income exceeds $50K per year based on various demographic and employment-related features using the Adult Census Income dataset.

## Project Overview

This project implements and compares multiple machine learning algorithms to predict employee salary brackets (<=50K or >50K) using demographic and employment data. The analysis includes comprehensive data preprocessing, exploratory data analysis, feature engineering, and model evaluation.

## Dataset

The project uses the **Adult Census Income Dataset** (also known as "Census Income" dataset) which contains demographic information extracted from the 1994 Census database.

### Features

- **age**: Age of the individual
- **workclass**: Type of employment (Private, Self-emp-not-inc, Local-gov, State-gov, Self-emp-inc, Federal-gov, etc.)
- **fnlwgt**: Final weight (number of people the census believes the entry represents)
- **education**: Highest level of education achieved
- **educational-num**: Number of years of education
- **marital-status**: Marital status
- **occupation**: Type of occupation
- **relationship**: Relationship status
- **race**: Race of the individual
- **gender**: Gender (Male/Female)
- **capital-gain**: Capital gains
- **capital-loss**: Capital losses
- **hours-per-week**: Hours worked per week
- **native-country**: Country of origin
- **income**: Target variable (<=50K or >50K)

### Dataset Statistics

- **Original size**: 48,842 records
- **After preprocessing**: 46,720 records
- **Features**: 14 (after removing target variable)

## Data Preprocessing

### 1. Missing Value Handling
- Replaced '?' values with 'Others' in categorical features (workclass, occupation)
- No null values found in the dataset

### 2. Data Cleaning
- Removed records with workclass values 'Without-pay' and 'Never-worked'
- Filtered age range to 17-75 years to remove outliers

### 3. Feature Engineering
- Applied Label Encoding to all categorical variables
- Standardized numerical features using StandardScaler

### 4. Outlier Detection
- Used boxplot visualization to identify and remove age outliers
- Removed extreme values outside the 17-75 age range

## Machine Learning Models

The project implements and compares five different classification algorithms:

### 1. Logistic Regression
- **Accuracy**: 81.49%
- **Precision (>50K)**: 69%
- **Recall (>50K)**: 46%
- **F1-Score (>50K)**: 55%

### 2. Random Forest Classifier
- **Accuracy**: 85.08%
- **Precision (>50K)**: 74%
- **Recall (>50K)**: 62%
- **F1-Score (>50K)**: 67%

### 3. K-Nearest Neighbors (KNN)
- **Accuracy**: 82.45%
- **Precision (>50K)**: 67%
- **Recall (>50K)**: 60%
- **F1-Score (>50K)**: 63%

### 4. Support Vector Machine (SVM)
- **Accuracy**: 83.96%
- **Precision (>50K)**: 75%
- **Recall (>50K)**: 54%
- **F1-Score (>50K)**: 63%

### 5. Gradient Boosting Classifier ⭐ **Best Model**
- **Accuracy**: 85.71%
- **Precision (>50K)**: 78%
- **Recall (>50K)**: 60%
- **F1-Score (>50K)**: 68%

## Model Performance Comparison

The **Gradient Boosting Classifier** achieved the best overall performance with:
- Highest accuracy (85.71%)
- Best precision for high-income prediction (78%)
- Strong F1-score balance (68%)

## Requirements

```
pandas
numpy
matplotlib
scikit-learn
```

## Installation

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Usage

1. Load the dataset:
```python
import pandas as pd
data = pd.read_csv('adult.csv')
```

2. Run the preprocessing pipeline:
```python
# Handle missing values
data.workclass.replace({'?':'Others'}, inplace=True)
data.occupation.replace({'?':'Others'}, inplace=True)

# Remove outliers
data = data[(data['age'] <= 75) & (data['age'] >= 17)]
```

3. Train and evaluate models:
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GradientBoostingClassifier())
])

# Train model
pipe.fit(X_train, y_train)

# Make predictions
predictions = pipe.predict(X_test)
```

## Key Insights

1. **Feature Importance**: Education level, age, hours per week, and occupation are strong predictors of income
2. **Class Imbalance**: The dataset shows more individuals earning <=50K (approximately 75%) than >50K (25%)
3. **Model Selection**: Ensemble methods (Random Forest, Gradient Boosting) outperform simpler models
4. **Preprocessing Impact**: Proper handling of categorical variables and outliers significantly improves model performance

## Visualizations

The project includes:
- Boxplots for outlier detection
- Model accuracy comparison bar charts
- Distribution analysis of key features

## Future Improvements

1. **Feature Engineering**: Create interaction features and polynomial features
2. **Hyperparameter Tuning**: Use GridSearchCV or RandomizedSearchCV for optimal parameters
3. **Handle Class Imbalance**: Apply SMOTE or class weights
4. **Feature Selection**: Use recursive feature elimination or feature importance analysis
5. **Cross-Validation**: Implement k-fold cross-validation for more robust evaluation
6. **Deep Learning**: Experiment with neural networks for potentially better performance

## Project Structure

```
.
├── employee salary prediction.ipynb  # Main Jupyter notebook
├── adult.csv                         # Dataset (not included)
└── README.md                         # This file
```

## License

This project uses the Adult Census Income dataset from the UCI Machine Learning Repository.

## Author

Data Science Project - Employee Salary Prediction

## Acknowledgments

- Dataset source: UCI Machine Learning Repository
- Census Bureau for the original data collection
