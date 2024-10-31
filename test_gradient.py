import GPy
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
X = np.random.uniform(-3., 3., (20, 1))
Y = np.sin(X) + 0.5 * np.random.randn(20, 1)

kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
model = GPy.models.GPRegression(X, Y, kernel)
model.optimize(messages=True)

# Plot the model
model.plot()
plt.title("GP Regression with GPy")

gradients,_ = model.predictive_gradients(X)

plt.figure()    
plt.scatter(X,gradients.flatten())
plt.title("Gradients of the Mean Function")
plt.xlabel("Index of Data Point")
plt.ylabel("Gradient Value")
plt.grid(True)
plt.show()
