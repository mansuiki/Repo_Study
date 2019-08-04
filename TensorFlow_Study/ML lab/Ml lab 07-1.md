# Lab 07-1
### MNIST!

```python
import tensorflow as tf
import matplotlib.pyplot as plt
import random

tf.set_random_seed(777)

from tensorflow.examples.tutorials.mnist import input_data

# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 10

# MNIST data image of shape 28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784])
# 0 - 9 digits recognition = 10 classes
Y = tf.placeholder(tf.float32, [None, nb_classes])

W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

# Hypothesis (using softmax)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Test model
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# parameters
num_epochs = 15
batch_size = 100
num_iterations = int(mnist.train.num_examples / batch_size)

with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    # Training cycle
    for epoch in range(num_epochs):
        avg_cost = 0

        for i in range(num_iterations):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, cost_val = sess.run([train, cost], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += cost_val / num_iterations

        print("Epoch: {:04d}, Cost: {:.9f}".format(epoch + 1, avg_cost))

    print("Learning finished")

    # Test the model using test sets
    print(
        "Accuracy: ",
        accuracy.eval(
            session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}
        ),
    )

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r : r + 1], 1)))
    print(
        "Prediction: ",
        sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r : r + 1]}),
    )

    plt.imshow(
        mnist.test.images[r : r + 1].reshape(28, 28),
        cmap="Greys",
        interpolation="nearest",
    )
    plt.show()
```

> ### 잉? 제대로 작동 안하네, 내 마음대로 해봐야지 ㅎㅎ

```python
# Keras 의 https://keras.io/examples/mnist_mlp/
# 길벗 출판사 의 https://github.com/gilbutITbook/006958

import os

import matplotlib.pyplot as plt
import numpy
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.datasets import mnist
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils import np_utils

seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# MNIST 데이터 불러오기
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 784).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 784).astype('float32') / 255

Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)

# 모델 프레임 설정 -> 수정수정수정!
model = Sequential()
model.add(Dense(512, input_dim=784, activation='relu'))
# Dropout 은 0.5 값을 보통 넘어가지 않아!
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
# Dropout -> 일정 비율만큼 연결을 끊어서 과한 학습을 방지!
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# 모델 실행 환경 설정
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 모델 최적화 설정
MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath = "./model/{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

# 모델의 실행
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=30, batch_size=200, verbose=0,
                    callbacks=[early_stopping_callback, checkpointer])

print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

# 그래프로 표현
x_len = numpy.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')

# 그래프에 그리드를 주고 레이블을 표시
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch') # 시행 횠수
plt.ylabel('loss') # Loss
plt.show()

'''
Epoch 00030: val_loss did not improve from 0.05586
   32/10000 [..............................] - ETA: 0s
  832/10000 [=>............................] - ETA: 0s
 1696/10000 [====>.........................] - ETA: 0s
 2528/10000 [======>.......................] - ETA: 0s
 3328/10000 [========>.....................] - ETA: 0s
 4096/10000 [===========>..................] - ETA: 0s
 4832/10000 [=============>................] - ETA: 0s
 5632/10000 [===============>..............] - ETA: 0s
 6464/10000 [==================>...........] - ETA: 0s
 7296/10000 [====================>.........] - ETA: 0s
 8128/10000 [=======================>......] - ETA: 0s
 8960/10000 [=========================>....] - ETA: 0s
 9792/10000 [============================>.] - ETA: 0s
10000/10000 [==============================] - 1s 63us/step
 Test Accuracy: 0.9841
'''
```

> #### CNN 버전은 조금 공부좀 더하고 해보자!