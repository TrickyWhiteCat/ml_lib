from keras.datasets import mnist
from logistic_regression import LogisticRegression
(x_train, y_train), (x_test, y_test) = mnist.load_data()
lr = LogisticRegression()
lr.x = x_train
lr.y = y_train
lr.lambda_ = 0.1
lr.iterate_num = 20
lr.scaling_method = 'standardize'
lr.fit()
print(lr.theta)