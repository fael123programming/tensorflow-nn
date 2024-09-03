import tensorflow as tf
import keras as kr
import numpy as np

# Model the linear function: y = 2x - 1.
# The model will predict, given an x, what the y value will be.

model = kr.Sequential([kr.layers.Dense(units=1, input_shape=[1])])  # Defining the model itself: single layer with single neuron with single value, the x.
model.compile(optimizer="sgd", loss="mean_squared_error")  # Compiling the model passing sgd (stochastic gradient descent) as the optimizer function, and mean_squared_error as the loss function.

# Data to fit the model.
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500)  # Fit the x's to the y's in 500 iterations.

print(model.predict(np.array([10.0])))