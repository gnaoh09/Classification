from __future__ import division, print_function, unicode_literals
import math
import numpy as np 
import matplotlib.pyplot as plt
def grad(x):
    return 2*x +10*np.cos(x)

def cost(x):
    return x**2 + 10*np.sin(x)

def myGD1(eta, x0):
    x = [x0]
    for it in range(1000):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) < 0.0001:
            break
        x.append(x_new)
    return (x, it)

def has_converged(theta_new):
    return np.linalg.norm(grad(theta_new)) < 1e-3

def GD_momentum(theta_init, eta, gamma):
    # Suppose we want to store history of theta
    theta = [theta_init]
    v_old = np.zeros_like(theta_init)
    for z in range(100):
        v_new = gamma*v_old + eta*grad(theta[-1])
        theta_new = theta[-1] - v_new
        if has_converged(theta_new):
            break 
        theta.append(theta_new)
        v_old = v_new
    return (theta[-1], z) 

(x1, it1) = myGD1(.1, 5)
(x2,z) = GD_momentum(5, 0.1,0.9 )
print('Solution x1 = %f, cost1 = %f, obtained after %d iterations'%(x1[-1], cost(x1[-1]), it1))
print('Solution x2 = %f, cost2 = %f, obtained after %d iterations'%(x2, cost(x2),z))  