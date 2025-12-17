# Health & Fitness Predictive Analytics Project

## 📌 Project Overview

This project applies **data preprocessing, exploratory data analysis (EDA), machine learning, and visualization techniques** to analyze health and fitness survey data.
The goal is to extract meaningful insights, identify behavioral patterns, and build predictive models related to fitness habits, mental well-being, and willingness to pay for fitness services.

The project is implemented entirely in **Python** using popular data science libraries.

---

## 🧠 Objectives

* Clean and preprocess raw survey data
* Perform exploratory data analysis (EDA)
* Identify correlations between health-related features
* Apply clustering techniques to group individuals
* Build predictive models for:

  * Exercise consistency
  * Mental well-being
  * Willingness to pay for fitness services
* Evaluate model performance using standard metrics

---
📂 Dataset Description (Primary Data)

Data Type: Primary data
Collection Method: Online survey (Google Forms)
File Format: CSV
File Name: _ Health & Fitness Insights - Form responses 1.csv

### Preprocessing Steps:

* Removed unnecessary columns (`Timestamp`, `Email address`)
* Cleaned column names
* Filled missing values using forward fill
* Encoded categorical variables using `LabelEncoder`
* Scaled features using `StandardScaler` where required

---

## 🛠️ Technologies & Libraries Used

* **Python**
* **pandas, numpy**
* **matplotlib, seaborn**
* **scikit-learn**

### Machine Learning Models:

* Decision Tree Classifier
* Logistic Regression
* K-Nearest Neighbors (KNN)
* Random Forest Classifier
* K-Means Clustering
* Principal Component Analysis (PCA)

---

## 📊 Exploratory Data Analysis (EDA)

The following analyses and visualizations are included:

* Dataset shape, data types, and statistical summary
* Missing value analysis
* Distribution of exercise consistency
* Correlation heatmap
* Feature distribution plots
* Pair plots for selected health indicators

---

## 📈 Clustering & Dimensionality Reduction

### 🔹 Elbow Method

Used to determine the optimal number of clusters for K-Means.

### 🔹 K-Means Clustering

* Groups individuals based on scaled health features
* Visualized using scatter plots

### 🔹 PCA (Principal Component Analysis)

* Reduces dimensionality to 2 components
* Visualizes clusters more effectively

---

## 🤖 Predictive Modeling & Evaluation

### 1️⃣ Decision Tree – Exercise Consistency

* Predicts exercise consistency
* Evaluated using **accuracy score**
* Decision tree visualization included

### 2️⃣ Logistic Regression – Mental Well-being

* Converts mental well-being into binary classification
* Evaluated using:

  * ROC Curve
  * AUC Score

### 3️⃣ Decision Tree – Willingness to Pay

* Evaluated using:

  * Confusion Matrix
  * Classification Report

### 4️⃣ Additional Models

* **KNN Classifier** – Accuracy evaluation
* **Random Forest Classifier** – 5-fold cross-validation accuracy

---

## 📌 Model Evaluation Metrics

* Accuracy Score
* ROC Curve
* AUC Score
* Confusion Matrix
* Classification Report
* Cross-validation accuracy

---

## 📌 Key Outcomes

* Identified meaningful patterns in health and fitness behavior
* Successfully clustered participants using unsupervised learning
* Built predictive models with reasonable accuracy
* Visualized insights for better interpretability

---
