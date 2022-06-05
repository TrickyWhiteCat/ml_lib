from keras.datasets import mnist
from models.LogisticRegression import LogisticRegression
import numpy as np
lr = LogisticRegression()
(x_train, y_train), (x_test, y_test) = mnist.load_data()
lr.set_x(x_train[:60000])
lr.set_y(y_train[:60000])
lr.use_gradient_descent(False)
lr.set_lambda(0.1)
lr.set_iter_num(100)
#lr.set_method('BFGS')
lr.set_scaling_method('normalize')
lr.disp = True
#import time
#start = time.time()
lr.fit()
#print('Time taken: {}'.format(time.time() - start))
correct = 0
for i in range(10000):
    if lr.predict(x_test[i]) == y_test[i]:
        correct += 1
print('Accuracy: {}'.format(correct / 10000))
#import random
#id = random.randint(0, 10000)
#print(lr.predict(x_test[id]), y_test[id])
#print(lr.cost())
#x = np.array([[0], [1], [4], [5]])
#y = np.array([2, 4, 10, 12])
#lr.set_x(x)
#lr.set_y(y)
#lr.set_iter_num(20)
#lr.set_lambda(0)
#lr.set_scaling_method('normalize')
#lr.fit()
#print(lr.predict([1]))