import tensorflow as tf
import numpy as np
from tensorflow import keras


'''
Fashion MNIST is composed of 70k 28x28 images of clothing so that
each one belongs to a class ranging from 0 to 9.
The train images are 60k images, around 85.71% of the total data and
the test images are 10k images, around 14.29% of the total data.

Label 	Class
0 	T-shirt/top
1 	Trouser
2 	Pullover
3 	Dress
4 	Coat
5 	Sandal
6 	Shirt
7 	Sneaker
8 	Bag
9 	Ankle boot

https://complex-valued-neural-networks.readthedocs.io/en/latest/code_examples/fashion_mnist.html
'''


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

model = keras.Sequential([  # Defining the model with 3 layers.
    keras.layers.Flatten(input_shape=(28, 28)),  # In the input layer, converting the images from matrix form to array.
    keras.layers.Dense(128, activation=tf.nn.relu),  # 128 neurons working as parameterized functions to predict the object. The ReLU (Rectified Linear Unit) activation function simply filter values greater than zero. 
    keras.layers.Dense(10, activation=tf.nn.softmax)  # The output comprises of 10 different classes. Only the biggest predicted value will be selected by the activation function softmax.
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy'
)

model.fit(train_images, train_labels, epochs=100)  # Fit the train images to the train labels in the given number of iterations.

evaluation = model.evaluate(test_images, test_labels)

print(f'Evaluation: {evaluation}')

img = keras.utils.load_img(
    './bag.png',
    color_mode='grayscale',
    target_size=(28, 28)
)

input_arr = np.array([keras.utils.img_to_array(img)])

predictions = model.predict(input_arr)

print(f'Prediction: {predictions}')  # The right answer should be 8, a bag.