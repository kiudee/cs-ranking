.. _intro:

|Build Status| |Coverage| |Binder|

************
Introduction
************
CS-Rank is a Python package for context-sensitive ranking and choice algorithms.

We implement the following new object ranking/choice architectures:

* FATE (First aggregate then evaluate)
* FETA (First evaluate then aggregate)   
 
In addition, we also implement these algorithms for choice functions:

* RankNetChoiceFunction
* GeneralizedLinearModel
* PairwiseSVMChoiceFunction

These are the state-of-the-art approaches implemented for the discrete choice setting:

* GeneralizedNestedLogitModel
* MixedLogitModel
* NestedLogitModel
* PairedCombinatorialLogit
* RankNetDiscreteChoiceFunction
* PairwiseSVMDiscreteChoiceFunction

Check out our `interactive notebooks`_ to quickly find out what our package can do.


Getting started
===============
As a simple "Hello World!"-example we will try to learn the Pareto problem:

.. code-block:: python

   import csrank as cs
   from csrank import ChoiceDatasetGenerator
   gen = ChoiceDatasetGenerator(dataset_type='pareto',
                                   n_objects=30,
                                   n_features=2)
   X_train, Y_train, X_test, Y_test = gen.get_single_train_test_split()                     
All our learning algorithms are implemented using the scikit-learn estimator API.
Fitting our FATENet architecture is as simple as calling the ``fit`` method:

.. code-block:: python

   fate = cs.FATEChoiceFunction(n_object_features=2)
   fate.fit(X_train, Y_train) 

Predictions can then be obtained using:

.. code-block:: python

   fate.predict(X_test)


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
You can cite our `arXiv papers`_::

  @article{csrank2019,
    author    = {Karlson Pfannschmidt and
                 Pritha Gupta and
                 Eyke H{\"{u}}llermeier},
    title     = {Learning Choice Functions: Concepts and Architectures },
    journal   = {CoRR},
    volume    = {abs/1901.10860},
    year      = {2019}
  }

  @article{csrank2018,
    author    = {Karlson Pfannschmidt and
                 Pritha Gupta and
                 Eyke H{\"{u}}llermeier},
    title     = {Deep architectures for learning context-dependent ranking functions},
    journal   = {CoRR},
    volume    = {abs/1803.05796},
    year      = {2018}
  }

License
--------
`Apache License, Version 2.0 <https://github.com/kiudee/cs-ranking/blob/master/LICENSE>`_

.. |Binder| image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/kiudee/cs-ranking/master?filepath=docs%2Fnotebooks

.. |Build Status| image:: https://travis-ci.org/kiudee/cs-ranking.svg?branch=master
   :target: https://travis-ci.org/kiudee/cs-ranking

.. |Coverage| image:: https://coveralls.io/repos/github/kiudee/cs-ranking/badge.svg
   :target: https://coveralls.io/github/kiudee/cs-ranking

.. _interactive notebooks: https://mybinder.org/v2/gh/kiudee/cs-ranking/master?filepath=docs%2Fnotebooks
.. _arXiv papers: https://arxiv.org/search/cs?searchtype=author&query=Pfannschmidt%2C+K
