.. _intro:

|Build Status| |Coverage| |Binder|

************
Introduction
************
CS-Rank is a Python package for context-sensitive ranking algorithms.

We implement the following new object ranking architectures:

* FATE (First aggregate then evaluate)
* FETA (First evaluate then aggregate)   
 
In addition we also offer these benchmark algorithms:

* Expected Rank Regression
* RankNet
* RankSVM

Check out our `interactive notebooks`_ to quickly find out what our package can do.


Getting started
===============
As a simple "Hello World!"-example we will try to learn the Medoid problem:

.. code-block:: python

   import csrank as cs
   from csrank import SyntheticDatasetGenerator
   gen = SyntheticDatasetGenerator(dataset_type='medoid',
                                   n_objects=5,
                                   n_features=2)
   X_train, Y_train, X_test, Y_test = gen.get_single_train_test_split()                     
All our learning algorithms are implemented using the scikit-learn estimator API.
Fitting our FATE-Network algorithm is as simple as calling the ``fit`` method:

.. code-block:: python

   fate = cs.FATEObjectRanker(n_object_features=2)
   fate.fit(X_train, Y_train) 

Predictions can then be obtained using:

.. code-block:: python

   fate.predict(X_test, Y_test)


Installation
------------
The latest release version of CS-Rank can be installed from Github as follows::

   pip install git+https://github.com/kiudee/cs-ranking.git

Another option is to clone the repository and install CS-Rank using::

   python setup.py install


Dependencies
------------
CS-Rank depends on Tensorflow, Keras, NumPy, SciPy, matplotlib, scikit-learn, scikit-optimize, joblib and tqdm.
For data processing and generation you will also need PyGMO, H5Py and pandas.


Citing CS-Rank
----------------
To be announced.

License
--------
`Apache License, Version 2.0 <https://github.com/kiudee/cs-ranking/blob/master/LICENSE>`_

.. |Binder| image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/kiudee/cs-ranking/master?filepath=notebooks

.. |Build Status| image:: https://travis-ci.org/kiudee/cs-ranking.svg?branch=master
   :target: https://travis-ci.org/kiudee/cs-ranking

.. |Coverage| image:: https://coveralls.io/repos/github/kiudee/cs-ranking/badge.svg
   :target: https://coveralls.io/github/kiudee/cs-ranking

.. _interactive notebooks: https://mybinder.org/v2/gh/kiudee/cs-ranking/master?filepath=notebooks