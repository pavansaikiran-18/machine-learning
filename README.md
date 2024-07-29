
### Credit Score Classification Project

This repository showcases a credit score classification project where we applied various machine learning algorithms to predict credit scores. The dataset was divided into training and test sets to train and evaluate the models. Each algorithm's performance was assessed using accuracy, F1 score, and other relevant metrics. Below is a detailed overview of the machine learning algorithms used and their evaluation metrics.

#### **Algorithms Applied:**

1. **Logistic Regression**
   - **Description:** Logistic Regression is a statistical model that uses a logistic function to model a binary dependent variable. It is widely used for binary classification tasks.
   - **Performance Metrics:**
     - **Accuracy:** Measures the proportion of correctly classified instances out of the total instances.
     - **F1 Score:** The harmonic mean of Precision and Recall, providing a balance between them.
     - **Precision:** The ratio of true positives to the sum of true positives and false positives.
     - **Recall:** The ratio of true positives to the sum of true positives and false negatives.

2. **Linear Discriminant Analysis (LDA)**
   - **Description:** LDA is a classification method that finds a linear combination of features that best separates two or more classes. It is used for dimensionality reduction and classification.
   - **Performance Metrics:**
     - **Accuracy:** The proportion of correctly predicted instances.
     - **F1 Score:** Balances Precision and Recall.
     - **Precision:** The accuracy of positive predictions.
     - **Recall:** The ability to find all positive instances.

3. **Support Vector Machine (SVM)**
   - **Description:** SVM is a supervised learning model that aims to find the hyperplane that best separates different classes in the feature space.
   - **Performance Metrics:**
     - **Accuracy:** The fraction of correctly classified instances.
     - **F1 Score:** The harmonic mean of Precision and Recall.
     - **Precision:** The proportion of positive identifications that were actually correct.
     - **Recall:** The ability of the classifier to find all positive samples.

4. **Decision Tree Classifier**
   - **Description:** A Decision Tree Classifier creates a model that predicts the value of a target variable based on several input features using a tree-like graph of decisions.
   - **Performance Metrics:**
     - **Accuracy:** The rate of correct predictions.
     - **F1 Score:** Balances the Precision and Recall.
     - **Precision:** Measures the correctness of the positive class.
     - **Recall:** Measures the ability to capture all positive instances.

5. **Random Forest Classifier**
   - **Description:** Random Forest is an ensemble learning method that combines multiple decision trees to improve classification accuracy and control overfitting.
   - **Performance Metrics:**
     - **Accuracy:** The overall correctness of the model.
     - **F1 Score:** Provides a single metric that balances Precision and Recall.
     - **Precision:** The correctness of positive predictions.
     - **Recall:** The completeness of the positive class predictions.

6. **Gradient Boosting Classifier**
   - **Description:** Gradient Boosting is an ensemble technique that builds models sequentially, each correcting the errors of the previous one to improve overall accuracy.
   - **Performance Metrics:**
     - **Accuracy:** The proportion of correct predictions.
     - **F1 Score:** The balance between Precision and Recall.
     - **Precision:** The proportion of true positive results in all positive predictions.
     - **Recall:** The ratio of correctly identified positive cases to the total number of positives.

7. **K-Nearest Neighbors (KNN)**
   - **Description:** KNN is a non-parametric method used for classification and regression by analyzing the distance between data points.
   - **Performance Metrics:**
     - **Accuracy:** Measures how often the classifier is correct.
     - **F1 Score:** The balance of Precision and Recall.
     - **Precision:** The accuracy of the positive class predictions.
     - **Recall:** The ability to identify all positive cases.

#### **Evaluation Metrics:**
- **Accuracy:** The ratio of the number of correct predictions to the total number of predictions. It is a general measure of model performance.
- **F1 Score:** A metric that combines Precision and Recall into a single score, especially useful when class distribution is imbalanced.
- **Precision:** Measures how many of the predicted positives are actually positive, focusing on the quality of positive predictions.
- **Recall:** Measures how many of the actual positives are captured by the model, focusing on the completeness of positive predictions.

The results of this project provide insights into the effectiveness of each algorithm for predicting credit scores. Each algorithm's performance was evaluated based on the above metrics, and the results are summarized in the output dataset, which includes columns for:
- `Credit Actual`: The true credit score.
- `Credit Predicted`: The predicted credit score by the model.

This detailed analysis helps in understanding which model performs best for credit score classification and guides further improvements.

---

