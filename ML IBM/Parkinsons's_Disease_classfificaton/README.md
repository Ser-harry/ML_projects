# Parkinson's Disease Classifier

This project is a machine learning classifier developed to detect Parkinson's disease using multiple models: K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Logistic Regression, Decision Tree, and Linear Regression (for regression metrics). Each model's performance was evaluated, and the results were compiled into tables for easy comparison.

## Project Overview

Parkinson's disease is a progressive disorder of the nervous system that impacts movement. Accurate early-stage classification is essential to providing effective treatment options. This project builds and compares various machine learning models to classify Parkinson's disease, highlighting the effectiveness of each model through performance metrics.

## Models Used

The following models were implemented and evaluated:

- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Decision Tree**
- **Linear Regression** (evaluated with regression metrics for comparative analysis)

## Dataset

The dataset includes numerous features relevant to Parkinson's disease. Data preprocessing and feature selection were performed to improve model performance and reliability.

## Results

### Classification Models

The table below summarizes the performance of each classification model in terms of Accuracy, Jaccard Index, F1 Score, and Log Loss (where applicable):

| Model                   | Accuracy | Jaccard Index | F1 Score | Log Loss |
|-------------------------|----------|---------------|----------|----------|
| Logistic Regression     | 94.87%   | 0.931        | 0.964    | 0.2327   |
| K-Nearest Neighbors     | 87.18%   | 0.828        | 0.906    | -        |
| Support Vector Machine  | 94.87%   | 0.933        | 0.966    | -        |
| Decision Tree           | 92.31%   | 0.900        | 0.947    | -        |

### Linear Regression Metrics

Linear Regression was evaluated for regression metrics as follows:

| Metric                   | Value   |
|--------------------------|---------|
| Mean Absolute Error (MAE)| 0.1977  |
| Mean Squared Error (MSE) | 0.0681  |
| R-squared (R²)           | 0.3331  |

## Getting Started

To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/parkinsons-disease-classifier.git
2. **Install dependencies:**
   Install the required Python packages using:
   ```bash
   pip install -r requirements.txt
## Conclusion

This project provides a comparative analysis of different algorithms for classifying Parkinson's disease. The results demonstrate the effectiveness of each model, with Logistic Regression and SVM yielding the highest accuracy and F1 scores. This comparative approach can guide further research and practical applications in the early detection of Parkinson's disease.

## License

This project is licensed under the MIT License.
   
