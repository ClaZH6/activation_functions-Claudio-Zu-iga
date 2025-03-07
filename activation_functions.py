import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=0)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def swish(x, beta=1):
    return x * sigmoid(beta * x)

def plot_activation(x, y, title):
    plt.figure()
    plt.plot(x, y, label=title)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid()
    plt.legend()
    plt.show()

def main():
    x = np.linspace(-10, 10, 100)
    
    y_sigmoid = sigmoid(x)
    y_tanh = tanh(x)
    y_relu = relu(x)
    y_leaky_relu = leaky_relu(x)
    y_elu = elu(x)
    y_swish = swish(x)
    
    plot_activation(x, y_sigmoid, "Sigmoid")
    plot_activation(x, y_tanh, "Tanh")
    plot_activation(x, y_relu, "ReLU")
    plot_activation(x, y_leaky_relu, "Leaky ReLU")
    plot_activation(x, y_elu, "ELU")
    plot_activation(x, y_swish, "Swish")

if __name__ == "__main__":
    main()
