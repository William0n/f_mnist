import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
fmnist = tf.keras.datasets.fashion_mnist


(train_images, train_labels), (test_images, test_labels) = fmnist.load_data()

# plotting to see 36 randomly selected training image
plt.figure(figsize=(10,10))
random_inds = np.random.choice(60000,36)
for i in range(36):
    plt.subplot(6,6,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    image_ind = random_inds[i]
    plt.imshow(np.squeeze(train_images[image_ind]), cmap=plt.cm.binary)
    plt.xlabel(train_labels[image_ind])
    
#### normalizing data 
train_images = tf.keras.utils.normalize(train_images, axis = 1)
test_images = tf.keras.utils.normalize(test_images, axis = 1)

# Normal sequential model with normal dense layers which use relu + softmax activations
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = tf.nn.relu),
    tf.keras.layers.Dense(128, activation = tf.nn.relu),
    tf.keras.layers.Dense(10, activation = tf.nn.softmax)
    
    ])
                          
model.compile(loss = "sparse_categorical_crossentropy",
              optimizer = "SGD",
              metrics = ['accuracy'])


history = model.fit(train_images, train_labels, epochs = 10, batch_size = 64)

results = model.evaluate(test_images, test_labels)

# Seeing what the prediction for the 500th image is 

testPrediction = model.predict(test_images[[500]])

prediction = np.argmax(testPrediction)
result = train_images[:, :, 0]
plt.imshow(np.reshape(test_images[500], (28,28)))

# Reshaping the data for convolution input layer 
#train_images = tf.keras.utils.normalize(train_images, axis = 1)
train_images = np.expand_dims(train_images, axis= -1)

#test_images = tf.keras.utils.normalize(test_images, axis = 1)
test_images = np.expand_dims(test_images, axis= -1)

# using a sequential model with 
model2 =  tf.keras.Sequential([
    
    tf.keras.layers.Conv2D(24, kernel_size = (3,3), activation = 'relu'),
    tf.keras.layers.MaxPool2D(pool_size = (2,2), strides= 2),
    
    tf.keras.layers.Conv2D(36, kernel_size = (3,3), activation = 'relu'),
    tf.keras.layers.MaxPool2D(pool_size = (2,2), strides= 2),
    
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = tf.nn.relu),
    tf.keras.layers.Dense(128, activation = tf.nn.relu),
    tf.keras.layers.Dense(10, activation = tf.nn.softmax)
    
    ])


model2.compile(loss = "sparse_categorical_crossentropy",
              optimizer = "adam",
              metrics = ['accuracy'])

history_cnn = model2.fit(train_images, train_labels, epochs = 10, batch_size = 64)

results = model2.evaluate(test_images, test_labels)



# Accuracy graphs  

plt.style.use('fivethirtyeight')

plt.plot(history.history['accuracy'])
plt.plot(history_cnn.history['accuracy'])
plt.legend(['Basic NN','CNN'], loc = 'lower right')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history_cnn.history['loss'])
plt.legend(['Basic NN','CNN'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()


