import numpy as np
import matplotlib.pyplot as plt

# Generate two random datasets for demonstration
np.random.seed(42)
dataset1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 100)
dataset2 = np.random.multivariate_normal([2, 2], [[1, 0], [0, 1]], 100)

# Combine the datasets and create labels
X = np.concatenate((dataset1, dataset2))
y = np.concatenate((np.zeros(len(dataset1)), np.ones(len(dataset2))))

# Randomly shuffle the data
shuffled_indices = np.random.permutation(len(X))
X = X[shuffled_indices]
y = y[shuffled_indices]

# Define the SGD classifier
learning_rate = 0.01
num_epochs = 100
weights = np.random.randn(2)
bias = np.random.randn(1)

# Perform SGD to find the decision boundary line
for epoch in range(num_epochs):
    for i in range(len(X)):
        linear_output = np.dot(X[i], weights) + bias
        y_pred = 1 / (1 + np.exp(-linear_output))
        
        gradient_w = (y_pred - y[i]) * X[i]
        gradient_b = (y_pred - y[i])
        
        weights -= learning_rate * gradient_w
        bias -= learning_rate * gradient_b

# Plot the scatter plots of the two datasets
plt.scatter(dataset1[:, 0], dataset1[:, 1], label='Dataset 1')
plt.scatter(dataset2[:, 0], dataset2[:, 1], label='Dataset 2')

# Plot the decision boundary line
x_values = np.linspace(-4, 6, 100)
y_values = -(weights[0] * x_values + bias) / weights[1]
plt.plot(x_values, y_values, color='red', linewidth=2, label='Decision Boundary')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Binary Classification with SGD')
plt.legend()
plt.show()