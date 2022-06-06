from keras.datasets import mnist
from models.models import LogisticRegression
import numpy as np
(x_train, y_train), (x_test, y_test) = mnist.load_data()
lr = LogisticRegression()
lr.set_x(x_train[:10000])
lr.set_y(y_train[:10000])
lr._use_gd = True
lr.set_lambda(0.1)
lr.set_num_iters(100)
lr.set_method('BFGS')
lr.set_scaling_method('normalize')
import time
start = time.time()
lr.fit()
print('Time taken: {}'.format(time.time() - start))
correct = 0
for i in range(10000):
    if lr.predict(x_test[i]) == y_test[i]:
        correct += 1
print('Accuracy: {}'.format(correct / 10000))
print('Train accuracy: {}'.format(lr.train_accuracy()))
