Introduction
-------------
CS-Rank is a Python package for context-sensitive ranking algorithms.


Getting started
---------------
As a simple "Hello World!"-example we will try to learn the Medoid problem:
```python
import csrank as cs
from csrank import SyntheticDatasetGenerator
gen = SyntheticDatasetGenerator(dataset_type='medoid')
X_train, Y_train, X_test, Y_test = gen.get_single_train_test_split()
```
All our learning algorithms are implemented using the scikit-learn estimator API.
Fitting our FATE-Network algorithm is as simple as calling the `fit` method:
```python
fate = cs.FATEObjectRanker()
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
