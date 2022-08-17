<h1 style = "color:red">models.KNearestNeighbors</h1>

**class KNearestNeighbor()**

*Create a K-nearest neighbor model. K-nearest neighbor is a model giving prediction based on K most similar (nearest) samples of a given input.*

<h3 style = 'color: navy'> Method </h3>

Method|Return|Description
--- |---|-----
`set_x(value:list-like)`|None|Set the input data.
`set_y(value:list-like)`|None|Set the target data.
`set_scaling_method(value:str)`|None|Set the scaling method. Supportted method: '_normalize_' (a.k.a _min-max scaling_), '_standardize_' or `None` (no scaling)
`set_k`|None|Set the number of _neighbors_ that the model will use to predict the result
`predict(x:list-like)`|tuple|Predict the label of the input. Return the class that the model _thinks_ is correct
