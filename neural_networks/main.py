import tensorflow as tf
from tensorflow import keras  # load data
import numpy as np
import matplotlib.pyplot as plt

# get data
data = keras.datasets.fashion_mnist

# subdivide data into training and test
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# labels (specified in the info section of the data)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0

model = keras.Sequential([  # Sequence of layers
    keras.layers.Flatten(input_shape=(28,28)),  # 28 x 28 pixel image
    keras.layers.Dense(128, activation="relu"),  # fully connected hidden layer with rectifying liner unit activation
    keras.layers.Dense(10, activation="softmax")  # fully connected output layer with all values adding up to 1
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=5)  # model will see this data 5 times to review already seen images

# test_loss, test_acc = model.evaluate(test_images, test_labels)
#
# print("Tested Acc:", test_acc)

prediction = model.predict(test_images)
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual:" + class_names[test_labels[i]])
    plt.title("Prediction:" + class_names[np.argmax(prediction[i])])
    plt.show()
