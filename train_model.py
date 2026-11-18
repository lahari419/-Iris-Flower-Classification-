
import pickle

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load Iris Dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Model
model = KNeighborsClassifier(n_neighbors=3)

# Train Model
model.fit(X_train, y_train)

# Test Model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy * 100, "%")

# Save Model
with open("iris_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model Saved Successfully!")
