import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cross_entropy_loss(y, y_hat):
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

def mini_batch_gradient_descent(X, y, theta, alpha, batch_size, num_epochs):
    for epoch in range(num_epochs):
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]

            y_hat = sigmoid(X_batch.dot(theta))
            J = cross_entropy_loss(y_batch, y_hat)

            gradient = X_batch.T.dot(y_hat - y_batch) / batch_size

            theta -= alpha * gradient

    return theta

def plot_decision_boundary(X, y, theta):
    # Get the minimum and maximum values of the features
    minX = np.min(X[:, 0])
    maxX = np.max(X[:, 0])
    minY = np.min(X[:, 1])
    maxY = np.max(X[:, 1])

    # Create a grid of points
    xx, yy = np.meshgrid(np.linspace(minX, maxX, 200), np.linspace(minY, maxY, 200))

    # Flatten the grid
    X_grid = np.c_[xx.ravel(), yy.ravel()]

    # Predict the labels for the grid points
    y_hat = sigmoid(X_grid.dot(theta))

    # Reshape the predictions back into a 2D array
    y_hat = y_hat.reshape(xx.shape)

    # Plot the decision boundary
    plt.contour(xx, yy, y_hat, levels=[0.5], colors='black')

    # Plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y)

    # Set the labels and title
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Decision Boundary')

    # Show the plot
    plt.show()

# Generate two random data sets
X1 = np.random.rand(100, 2)
X2 = np.random.rand(100, 2) + 2

# Combine the data sets
X = np.vstack((X1, X2))

# Create labels for the data sets
y = np.concatenate((np.zeros(100), np.ones(100)))

# Initialize the parameters
theta = np.zeros(X.shape[1])

# Set the hyperparameters
alpha = 0.01
batch_size = 32
num_epochs = 1000

# Train the model
theta = mini_batch_gradient_descent(X, y, theta, alpha, batch_size, num_epochs)

# Visualize the decision boundary
plot_decision_boundary(X, y, theta)
