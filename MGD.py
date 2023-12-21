import numpy as np
import matplotlib.pyplot as plt

# Generate random data
np.random.seed(0)
X1 = np.random.randn(100, 2) + np.array([-2, -2])
X2 = np.random.randn(100, 2) + np.array([2, 2])
X = np.vstack((X1, X2))
y = np.concatenate((np.zeros(100), np.ones(100)))

# Add bias term to feature matrix
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define cross-entropy loss function
def cross_entropy_loss(y, y_hat):
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

# Define mini-batch gradient descent function
def mini_batch_gradient_descent(X, y, alpha, batch_size, num_epochs):
    theta = np.zeros(X.shape[1])  # Initialize parameters
    losses = []  # Track loss values during training

    for epoch in range(num_epochs):
        for i in range(0, X.shape[0], batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]

            y_hat = sigmoid(X_batch.dot(theta))
            loss = cross_entropy_loss(y_batch, y_hat)
            gradient = X_batch.T.dot(y_hat - y_batch) / batch_size

            theta -= alpha * gradient

        losses.append(loss)

    return theta, losses

# Set hyperparameters
alpha = 0.01
batch_size = 32
num_epochs = 100

# Train the model
theta, losses = mini_batch_gradient_descent(X, y, alpha, batch_size, num_epochs)

# Plot the decision boundary
x_values = np.linspace(-6, 6, 100)
y_values = -(theta[0] + theta[1] * x_values) / theta[2]

# Plot the data points
plt.scatter(X1[:, 0], X1[:, 1], label='Class 0')
plt.scatter(X2[:, 0], X2[:, 1], label='Class 1')

# Plot the decision boundary line
plt.plot(x_values, y_values, color='red', label='Decision Boundary')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Binary Classification with Mini-Batch Gradient Descent')
plt.legend()
plt.show()

