[![Build Status](https://travis-ci.org/kiudee/cs-ranking.svg?branch=master)](https://travis-ci.org/kiudee/cs-ranking)
[![Coverage Status](https://coveralls.io/repos/github/kiudee/cs-ranking/badge.svg)](https://coveralls.io/github/kiudee/cs-ranking)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/kiudee/cs-ranking/master?filepath=notebooks)

Introduction
-------------
CS-Rank is a Python package for context-sensitive ranking algorithms.

We implement the following new object ranking architectures:

 * FATE (First aggregate then evaluate)
 * FETA (First evaluate then aggregate)
 
In addition we also offer these benchmark algorithms:

* Expected Rank Regression
* RankNet
* RankSVM

Check out our [interactive notebooks](https://mybinder.org/v2/gh/kiudee/cs-ranking/master?filepath=notebooks) to quickly find out what our package can do.


Getting started
---------------
As a simple "Hello World!"-example we will try to learn the Medoid problem:
```python
import csrank as cs
from csrank import SyntheticDatasetGenerator
gen = SyntheticDatasetGenerator(dataset_type='medoid',
                                n_objects=5,
                                n_features=2)
X_train, Y_train, X_test, Y_test = gen.get_single_train_test_split()
```
All our learning algorithms are implemented using the scikit-learn estimator API.
Fitting our FATE-Network algorithm is as simple as calling the `fit` method:
```python
fate = FATEObjectRanker(n_object_features=2)
fate.fit(X_train, Y_train)
```
Predictions can then be obtained using:
```python
fate.predict(X_test, Y_test)
```

Installation
------------
The latest release version of CS-Rank can be installed from Github as follows:
```
pip install git+https://github.com/kiudee/cs-ranking.git
```
Another option is to clone the repository and install CS-Rank using 
```
python setup.py install
```

Dependencies
------------
CS-Rank depends on Tensorflow, Keras, NumPy, SciPy, matplotlib, scikit-learn, scikit-optimize, joblib and tqdm.
For data processing and generation you will also need PyGMO, H5Py and pandas.


Citing CS-Rank
----------------
To be announced.

License
--------
[Apache License, Version 2.0](https://github.com/kiudee/cs-ranking/blob/master/LICENSE)
