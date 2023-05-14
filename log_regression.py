import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

cancer_data = pd.read_csv('C:/Users/nickd/Downloads/archive/data.csv')

# print(cancer_data.head())

cancer_data = cancer_data.dropna(axis=1)
# print(cancer_data['diagnosis'])
labelencoder = LabelEncoder()
cancer_data['diagnosis'] = labelencoder.fit_transform(cancer_data['diagnosis'])
# print(cancer_data['diagnosis'])

# sns.pairplot(cancer_data, hue='diagnosis')

X = cancer_data.iloc[:, 2:31].values
Y = cancer_data.iloc[:, 1].values


train_accs, test_accs = [], []
for i in range(30000):
    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=i)

    # Scale the features
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)

    # Create the logistic regression classifier object
    clf = LogisticRegression()

    # Train the classifier on the training data
    clf.fit(X_train, Y_train)

    # Predict the labels for the training and testing data
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    # Compute the accuracy of the classifier on the training and testing data
    train_acc = accuracy_score(Y_train, y_pred_train)
    test_acc = accuracy_score(Y_test, y_pred_test)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    print(i)

# Compute the average training and testing accuracies
train_acc_avg = np.mean(train_accs)
test_acc_avg = np.mean(test_accs)

# Print the average training and testing accuracies
print("Average Train Accuracy: {:.2f}%".format(train_acc_avg * 100))
print("Average Test Accuracy: {:.2f}%".format(test_acc_avg * 100))

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Plot the training accuracy on the first subplot
ax1.plot(train_accs)
ax1.set_title('Training Accuracy')
ax1.set_xlabel('Random State')
ax1.set_ylabel('Accuracy')

# Plot the testing accuracy on the second subplot
ax2.plot(test_accs)
ax2.set_title('Testing Accuracy')
ax2.set_xlabel('Random State')
ax2.set_ylabel('Accuracy')

# Display the plot
plt.tight_layout()
plt.show()
