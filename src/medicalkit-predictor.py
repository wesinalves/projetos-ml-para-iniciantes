# load data set
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

data = pd.read_csv('../datasets/dataset/train.csv')

data = data.drop(data[data['Target'] == 0].sample(frac=0.8, random_state=42).index)
#print(data['Target'].value_counts())

# Preprocess the data
X = data.drop(["Gender", "ID", "Target"], axis=1)
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = SVC(kernel="rbf", C=10.0, gamma='scale')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = 100*f1_score(y_test, y_pred, average='weighted')
print(f"Model F1 score: {f1}")

# using model on test.csv ensure all the id columns are same
# test_data = pd.read_csv('../datasets/dataset/test.csv')
# X_final_test = test_data.drop(["Gender", "ID"], axis=1)
# X_final_test = scaler.transform(X_final_test)
# final_predictions = model.predict(X_final_test)
# output = pd.DataFrame({'ID': test_data['ID'], 'Target': final_predictions})
# output.to_csv('submission.csv', index=False)
