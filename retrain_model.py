import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model, tree, ensemble
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
import joblib

# Load the PCA reduced data
data = pd.read_csv('Final_Dataset/With_dimensionality_reduction/pca_final.csv')
data.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')  # Drop index if present

# Assuming the last column is target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the ensemble model as in the notebook
eclf = VotingClassifier(estimators=[
    ('SVM', SVC(probability=True, C=10, gamma=0.01)),  # Using best params from notebook
    ('knn', KNeighborsClassifier(n_neighbors=1)),
    ('rf', ensemble.RandomForestClassifier(n_estimators=300)),
    ('bayes', GaussianNB()),
    ('logistic', LogisticRegression(C=1.0, penalty='l2'))
], voting='soft')

# Fit the model
eclf.fit(X_train, y_train)

# Evaluate
y_pred = eclf.predict(X_test)
print("Accuracy:", eclf.score(X_test, y_test))
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(eclf, 'Deployment/Heart_Disease_Prediction_using_ECG (4).pkl')
print("Model saved as 'Deployment/Heart_Disease_Prediction_using_ECG (4).pkl'")