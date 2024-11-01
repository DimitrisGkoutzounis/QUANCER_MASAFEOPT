import GPy
import numpy as np
import matplotlib.pyplot as plt

# Generate some toy data
np.random.seed(42)
X = np.random.uniform(-3., 3., (20, 1))
Y = np.sin(X) + 0.5 * np.random.randn(20, 1)

plt.figure()
# Define and fit the GP model
kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
model = GPy.models.GPRegression(X, Y, kernel)
model.plot()


plt.figure()
mean, _ = model.predict(X)
grad_mean = model.kern.gradients_X(np.ones_like(mean), X, model.X)

# Plot the gradient of the mean function
plt.plot(grad_mean, 'r-', label='Gradient of Mean Function')
plt.xlabel("Input")
plt.ylabel("Gradient Value")
plt.legend()
plt.grid(True)
plt.show()
