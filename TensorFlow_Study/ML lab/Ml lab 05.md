# ML Lab 05
### Logistic Classification(Logistic Regression) in TensorFlow
-------
![img](img/lab05-1.png) 
> ### 연습해보자!
> * CSV 를 tf.decode_csv 로 읽어보자!
> * DataSet : https://www.kaggle.com/

```python
import tensorflow as tf

tf.set_random_seed(777)

x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Sigmoid 함수를 이용한 Hypothesis
# Sigmoid 함수를 이용하지 않고도 직접 구현할 수 있음 :  tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# hypothesis 값은 0.8, 0.2 ... 이런 식으로 나옴
# 이를 tf.cast 를 사용해 0.5 이상이면 1.0으로 아니면 0.0 으로 바꿔줌
# tf.cast(Boolean 판변식, dtype=결과타입)
# 즉 hypothesis > 0.5 가 True 이면 1.0을 반환함
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# 실제 Y 와 hypothesis 를 비교하여 적중률 계산
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val)

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)

'''
9000 0.16165249
9200 0.15906553
9400 0.1565599
9600 0.15413195
9800 0.1517783
10000 0.1494956

Hypothesis:  [[0.03074026]
 [0.15884683]
 [0.3048674 ]
 [0.78138196]
 [0.93957496]
 [0.9801688 ]] 
Correct (Y):  [[0.]
 [0.]
 [0.]
 [1.]
 [1.]
 [1.]] 
Accuracy:  1.0
'''
```

```python
import numpy as np
import tensorflow as tf

tf.set_random_seed(777)

# 파일 읽기
# CSV 를 tf.decode_csv 로 읽어보자!
# DataSet : https://www.kaggle.com/
xy = np.loadtxt('DeepLearningZeroToAll-master/data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, y_data.shape)

X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([8, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val)

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)

'''
Accuracy:  0.7628459
'''
```