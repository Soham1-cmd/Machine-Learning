import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the dataset
dataset = pd.read_csv("/Users/soham/Desktop/Machine Learning/iris.csv")

# Prepare the data
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train KNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict on test set and evaluate
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print("Accuracy:", accuracy)

# Find the best k value
Ks = 50
means_acc = np.zeros((Ks-1))
for n in range(1, Ks):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    yhat = knn.predict(X_test)
    means_acc[n-1] = accuracy_score(y_test, yhat)

best_accuracy = means_acc.max()
best_k = means_acc.argmax() + 1
print("The best accuracy was with", best_accuracy, "with k=", best_k)

# Plot accuracy vs. number of neighbors
plt.plot(range(1, Ks), means_acc, 'g')
plt.legend(("Accuracy",))
plt.ylabel("Accuracy")
plt.xlabel("Number of neighbors (k)")
plt.title("Accuracy vs. Number of Neighbors (K)")
plt.tight_layout()
plt.show()