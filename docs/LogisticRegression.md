<h1 style = "color:red">models.LogisticRegression</h1>

**class LogisticRegession()**

*Create a logistic regression model. The model will use one vs all method if there are more than two classes.*

<h3 style = 'color: navy'> Method </h3>

Method|Return|Description
--- |---|-----
`set_x(value:list-like)`|None|Set the input data.
`set_y(value:list-like)`|None|Set the target data.
`set_lambda(value:float\|int)`|None|Set the regularization parameter.
`set_method(value:str)`|None|Set the method of optimization. Default: None
`set_learning_rate(value: float|int)`|None|Set the learning rate for the model. Only needed when using Gradient Descent.
`set_num_iters(value:int)`|None|Set the maximum number of iterations. If not being set, there will be no limit.
`set_scaling_method(value:str)`|None|Set the scaling method. Supportted method: '_normalize_' (a.k.a _min-max scaling_), '_standardize_' or `None` (no scaling)
`fit()`|None|Fit the model.
`predict(x:list-like)`|tuple|Predict the label of the input. Return the class that the model _thinks_ is correct
