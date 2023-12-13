from __future__ import print_function
import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from cvxopt import matrix, solvers
np.random.seed(22)

means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N) # class 1
X1 = np.random.multivariate_normal(means[1], cov, N) # class -1 
X = np.concatenate((X0.T, X1.T), axis = 1) # all data 
y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1) # labels 

# build K
V = np.concatenate((X0.T, -X1.T), axis = 1)
K = matrix(V.T.dot(V)) # see definition of V, K near eq (8)

p = matrix(-np.ones((2*N, 1))) # all-one vector 
# build A, b, G, h 
G = matrix(-np.eye(2*N)) # for all lambda_n >= 0
h = matrix(np.zeros((2*N, 1)))
A = matrix(y) # the equality constrain is actually y^T lambda = 0
b = matrix(np.zeros((1, 1))) 
solvers.options['show_progress'] = False
sol = solvers.qp(K, p, G, h, A, b)

l = np.array(sol['x'])
print('lambda = ')
print(l.T)

epsilon = 1e-6 # just a small number, greater than 1e-9
S = np.where(l > epsilon)[0]

VS = V[:, S]
XS = X[:, S]
yS = y[:, S]
lS = l[S]
# calculate w and b
w = VS.dot(lS)
b = np.mean(yS.T - w.T.dot(XS))

print('w = ', w.T)
print('b = ', b)

# Plotting the scatter and decision boundary
plt.scatter(X0[:, 0], X0[:, 1], color='blue', label='Class 1')
plt.scatter(X1[:, 0], X1[:, 1], color='red', label='Class -1')
plt.xlabel('X1')
plt.ylabel('X2')

# Generate a grid of points in the feature space and classify them
x1_range = np.linspace(min(X[:, 0]) - 1, max(X[:, 0]) + 1, 100)
x2_range = np.linspace(min(X[:, 1]) - 1, max(X[:, 1]) + 1, 100)
xx1, xx2 = np.meshgrid(x1_range, x2_range)
X_grid = np.vstack((xx1.flatten(), xx2.flatten()))
y_grid = np.sign(w.T.dot(X_grid) + b).reshape(xx1.shape)

# Plot the decision boundary
plt.contour(xx1, xx2, y_grid, colors='black', levels=[-1, 0, 1], linestyles="solid")
plt.legend()
plt.show()