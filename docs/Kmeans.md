<h1 style = "color:red">models.KMeans</h1>

**class KMeans()**

*Create a K-means model. K-means is a unsupervised model used to split a dataset into K clusters based on their similarity.*

<h3 style = 'color: navy'> Method </h3>

Method|Return|Description
--- |---|-----
`set_x(value:list-like)`|None|Set the input data.
`set_num_centroids(value: int)`|None|Set the number of centroids
`set_num_iters(value:int)`|None|Set the maximum number of iterations. On each iteration, the model will randomly choose K samples to be initial centroids. As a result, K-means is prone to overfitting, therefore we need to randomly choosing initial centroids several times and choosing the one with the best performace.
`set_scaling_method(value:str)`|None|Set the scaling method. Supportted method: '_normalize_' (a.k.a _min-max scaling_), '_standardize_' or `None` (no scaling)
`fit()`|None|Fit the model.
`centroids`|np.ndarray|return a array containing values of centroids
