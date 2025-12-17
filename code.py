# ---------------------------------------------------------
# Health & Fitness Predictive Analytics Project
# ---------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, r2_score, confusion_matrix,
    classification_report, roc_curve, roc_auc_score
)

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# ---------------------------------------------------------
# FILE PATH
# ---------------------------------------------------------

file_path = r"C:\Users\deept\OneDrive\Desktop\int 234 project\_ Health & Fitness Insights - Form responses 1.csv"

# ---------------------------------------------------------
# DATA PREPROCESSING
# 

df = pd.read_csv(file_path)

df.drop(columns=["Timestamp", "Email address"], inplace=True, errors="ignore")
df.columns = [col.strip().split("?")[0] for col in df.columns]
df.ffill(inplace=True)

encoder = LabelEncoder()
for col in df.columns:
    df[col] = encoder.fit_transform(df[col])

print("\nPreprocessed Dataset Preview:")
print(df.head())

# =========================================================
# EDA – EXPLORATORY DATA ANALYSIS
# =========================================================

print("\nDataset Shape:", df.shape)
print("\nData Types:")
print(df.dtypes)

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values Per Column:")
print(df.isnull().sum())

# Target variable distribution (Exercise Consistency)
plt.figure(figsize=(6,4))
sns.countplot(x=df[df.columns[0]])
plt.title("Distribution of Exercise Consistency")
plt.xlabel("Exercise Consistency (Encoded)")
plt.ylabel("Count")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Feature Distributions (Key Health Indicators)
key_features = df.columns[:6]

plt.figure(figsize=(14,8))
for i, col in enumerate(key_features, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[col], kde=True)
    plt.title(col)

plt.tight_layout()
plt.show()

# Pairplot (Optional – comment if dataset is large)
sns.pairplot(df[key_features])
plt.show()

# =========================================================
# VISUALIZATION 1 – ELBOW METHOD
# =========================================================

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(scaled_data)
    wcss.append(km.inertia_)

plt.figure()
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method for Optimal Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

# =========================================================
# OBJECTIVE – K-MEANS CLUSTERING
# =========================================================

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

plt.figure()
plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=clusters)
plt.title("K-Means Cluster Visualization")
plt.xlabel("Feature 1 (Scaled)")
plt.ylabel("Feature 2 (Scaled)")
plt.show()

# =========================================================
# OBJECTIVE – PCA VISUALIZATION
# =========================================================

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

plt.figure()
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters)
plt.title("PCA Visualization of Fitness Clusters")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

# =========================================================
# VISUALIZATION – DECISION TREE (EXERCISE CONSISTENCY)
# =========================================================

X = df[[df.columns[12], df.columns[7], df.columns[13], df.columns[9]]]
y = df[df.columns[0]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model.fit(X_train, y_train)

print("\nDecision Tree Accuracy (Exercise Consistency):",
      accuracy_score(y_test, dt_model.predict(X_test)))

plt.figure(figsize=(18, 9))
plot_tree(dt_model, feature_names=X.columns, filled=True, rounded=True)
plt.title("Decision Tree – Exercise Consistency")
plt.show()

# =========================================================
# VISUALIZATION – ROC CURVE (MENTAL WELL-BEING)
# =========================================================

mental_col = df.columns[13]
df["Mental_Wellbeing_Binary"] = df[mental_col].apply(lambda x: 1 if x > 0 else 0)

X = df[[df.columns[12], df.columns[7], df.columns[9], df.columns[6]]]
y = df["Mental_Wellbeing_Binary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_prob = log_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Mental Well-being Impact")
plt.legend()
plt.grid(True)
plt.show()

print("AUC Score (Mental Well-being):", auc_score)

# =========================================================
# VISUALIZATION – CONFUSION MATRIX (WILLINGNESS TO PAY)
# =========================================================

X = df.drop(columns=[df.columns[-1]])
y = df[df.columns[-1]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

dt_pay = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_pay.fit(X_train, y_train)

y_pred = dt_pay.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Willingness to Pay")
plt.show()

print("\nClassification Report – Willingness to Pay:")
print(classification_report(y_test, y_pred, zero_division=0))

# =========================================================
# ADDITIONAL OBJECTIVES
# =========================================================

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print("KNN Accuracy:", accuracy_score(y_test, knn.predict(X_test)))

rf = RandomForestClassifier(random_state=42)
cv_scores = cross_val_score(rf, X, y, cv=5)
print("Random Forest CV Accuracy:", cv_scores.mean())

# =========================================================
# END OF PROJECT
# =========================================================











