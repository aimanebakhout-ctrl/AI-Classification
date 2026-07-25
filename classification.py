# Week 2 Project
# Data Classification using Decision Tree on Iris Dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
iris = load_iris()

# Features and labels
X = iris.data
y = iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)


# Make predictions
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model trained successfully!")
print("Predictions:", predictions)
print("Accuracy:", accuracy)