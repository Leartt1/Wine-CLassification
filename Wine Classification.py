import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the dataset
dataset = pd.read_csv("wine-data.csv")

# Extract the features (X) and dependent variable (Y)
X = dataset.iloc[:, 1:].values
Y = dataset.iloc[:, 0].values

# Map class labels to start from 0
unique_labels = np.unique(Y)
label_mapping = {label: i for i, label in enumerate(unique_labels)}
Y = np.array([label_mapping[label] for label in Y])

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Train the XGBoost classifier
classifier = XGBClassifier()
classifier.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred = classifier.predict(X_test)

# Compute confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
class_names = unique_labels.astype(str)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

# Print confusion matrix
print("Confusion Matrix:")
print(cm_df)

# Compute accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)

# Print predicted and actual labels
print(np.concatenate((Y_pred.reshape(len(Y_pred), 1), Y_test.reshape(len(Y_test), 1)), 1))
