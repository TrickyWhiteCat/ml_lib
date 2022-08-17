<h1 style = "color:red">models.LinearRegression</h1>

**class LinearRegession()**

*Create a linear regression model. A linear regression model is simply a function* $y=a_0+a_1x_1+ a_2x_2 + ... + a_nx_n$ *that best fit to the dataset*

<h3 style = 'color: navy'> Method </h3>

Method|Return|Description
--- |---|-----
`set_x(value:list-like)`|None|Set the input data.
`set_y(value:list-like)`|None|Set the target data.
`set_lambda(value:float|int)`|None|Set the regularization parameter.
`set_method(value:str)`|None|Set the method of optimization. Default: None
`set_learning_rate(value: float|int)`|None|Set the learning rate for the model. Only needed when using Gradient Descent.
`set_num_iters(value:int)`|None|Set the maximum number of iteration. If not being set, there will be no limit.
`set_scaling_method(value:str)`|None|Set the scaling method. Supportted method: '_normalize_' (a.k.a _min-max scaling_), '_standardize_' or `None` (no scaling)
`fit()`|None|Fit the model.
`predict(x:list-like)`|tuple|Predict the outcome of a given input.
