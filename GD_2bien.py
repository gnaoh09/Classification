import matplotlib.pyplot as plt
from random import random

def hypothesis(x, theta_0, theta_1):
    return x*theta_1 + theta_0

def compute_cost(theta_0, theta_1, xs, ys):
    cost = 0
    for i in range(len(xs)):
        error = hypothesis(xs[i], theta_0, theta_1) - ys[i]
        cost = cost + error*error
    return cost/(2*len(xs))

def cal_gradient_1(theta_0, theta_1, xs, ys):
    grad = 0
    for i in range(len(xs)):
        grad = grad + (hypothesis(xs[i], theta_0, theta_1) - ys[i])*xs[i]
    return grad/len(xs)

def cal_gradient_0(theta_0, theta_1, xs, ys):
    grad = 0
    for i in range(len(xs)):
        grad = grad + (hypothesis(xs[i], theta_0, theta_1) - ys[i])
    return grad/len(xs)

if __name__ == "__main__":
    xs = [3, 5, 3.25, 1.5]
    ys = [1.5, 2.25, 1.625, 1.0]

    alpha = 0.1

    theta_0 = 2*random() - 1
    theta_1 = 2*random() - 1
    gradient_0 = cal_gradient_0(theta_0, theta_1, xs, ys)
    gradient_1 = cal_gradient_1(theta_0, theta_1, xs, ys)

    cycle = 1
    while abs(gradient_0) > 0.0005 or abs(gradient_1) > 0.0005 :
        print("====CYCLE {}====".format(cycle))
        print("Theta 0: {}, Theta 1: {}".format(theta_0,theta_1))
        print("Gradient 0: {}, Gradient 1: {}".format(gradient_0,gradient_1))
        print("Cost: {}".format(compute_cost(theta_0, theta_1, xs, ys)))
        cycle = cycle + 1
        theta_0 = theta_0 - alpha*gradient_0
        theta_1 = theta_1 - alpha*gradient_1
        gradient_0 = cal_gradient_0(theta_0, theta_1, xs, ys)
        gradient_1 = cal_gradient_1(theta_0, theta_1, xs, ys)

    print("----------------------")
    print("FOUND THETA 0: {}, THETA 1: {} IS OPTIMUM".format(theta_0,theta_1))
    print("COST: {}".format(compute_cost(theta_0,theta_1, xs, ys)))

