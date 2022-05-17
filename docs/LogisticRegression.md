<h1 style = "color:red">model.lr.LogisticRegession</h1>

**class model.lr.LogisticRegession()**

*Create a logistic regression model. The model will use one vs all method if there are more than two classes.*

<h3 style = 'color: navy'> Method </h3>

Method|Return|Description
--- |---|-----
`set_x(value:list-like)`|None|Set the input data.
`set_y(value:list-like)`|None|Set the target data.
`set_lambda(value:float|int)`|None|Set the regularization parameter.
`set_method(value:str)`|None|Set the method of optimization.
`use_gradient_decent(value:bool)`|None|Whether or not to use gradient decent.
`set_iter_num(value:int)`|None|Set the maximum number of iteration. If not being set, there will be no limit.
`set_scaling_method(value:str)`|None|Set the scaling method. Either 'standardize' or 'normalize'.
`set_disp(value:bool)`|None|Whether or not to display the process.
`fit()`|None|Fit the model.
`predict(x|list-like)`|tuple|Predict the target of the input. Return the predicted class and a list of probability of each class.
`cost()`|ndarray|Return the cost of the model.