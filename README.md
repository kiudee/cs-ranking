Introduction
-------------
CS-Rank is a Python package for context-sensitive ranking algorithms.


Getting started
---------------
As a simple "Hello World!"-example we will try to learn the Medoid problem:
```python
import csrank as cs

fate = cs.FATEObjectRanker()
fate.fit(X, Y)
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
