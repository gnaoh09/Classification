import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.datasets import make_classification

# Generate a sample dataset
X, y = make_classification(n_samples=1000, n_features=3, n_redundant=0, n_informative=3,
                           n_clusters_per_class=1, random_state=42)

# Split into train and test sets
X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

# Manually implement logistic regression

# Add a column of ones to X_train for the bias term
X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1)

# Initialize weights with zeros
weights = np.zeros(X_train.shape[1])

# Set learning rate and number of iterations
learning_rate = 0.01
num_iterations = 1000

# Gradient descent
for _ in range(num_iterations):
    # Calculate the logits (predicted probabilities)
    logits = np.dot(X_train, weights)
    
    # Apply sigmoid function to obtain probabilities
    probabilities = 1 / (1 + np.exp(-logits))
    
    # Calculate the gradient
    gradient = np.dot(X_train.T, (probabilities - y_train)) / y_train.shape[0]
    
    # Update weights
    weights -= learning_rate * gradient

# Make predictions on the test set
X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1)
logits = np.dot(X_test, weights)
y_pred = np.round(1 / (1 + np.exp(-logits)))

# Evaluate accuracy
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot training data points
ax.scatter(X_train[:, 1], X_train[:, 2], X_train[:, 3], c=y_train, s=60, cmap='Set1')

# Plot test data points
ax.scatter(X_test[:, 1], X_test[:, 2], X_test[:, 3], c=y_test, s=60, cmap='Set1', alpha=0.5)

# Define the meshgrid for plotting the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))

# Compute the hyperplane values for each point on the meshgrid
zz = (-weights[0] - weights[1] * xx - weights[2] * yy) / weights[3]

# Plot the hyperplane as a 3D contour
ax.contour3D(xx, yy, zz, 100, colors='b', alpha=0.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()