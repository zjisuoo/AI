import numpy as np
import matplotlib.pyplot as plt

def NN(m1, m2, w1, w2, b):
    z = m1 * w1 + m2 * w2 + b
    return sigmoid(z)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

w1 = np.random.randn(10)
w2 = np.random.randn(10)
b = np.random.randn(10)

print(w1,"\n",w2,"\n",b,"\n")

a = NN(1, 1, w1, w2, b)
print(a)

plt.scatter(w1, w2)
plt.show()