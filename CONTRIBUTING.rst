============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps. Contributions do not have to be source code. There are many ways to
contribute:

Report Bugs
===========


Report bugs at the `issue tracker`_.

.. _issue tracker:
    https://github.com/kiudee/cs-ranking/issues

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.
* Ideally a `minimal reproducible example`__.

__ https://stackoverflow.com/help/minimal-reproducible-example

Submit Feedback
===============

Feedback can be submitted at the `issue tracker`_ as well.

.. _issue tracker:
    https://github.com/kiudee/cs-ranking/issues

If you are proposing a feature:

* Explain in detail how it would work and why you would want it.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that code contributions
  are welcome :)

Write Documentation
===================

This project could always use more documentation, whether as part of the
official cs-ranking docs, in docstrings, or even on the web in blog posts,
articles, and such. For writing in-project documentation, the setup is similar
to contributing code.

Contribute Code
===============

Ready to get your hands dirty?
Here's how to set up ``cs-ranking`` for local development.

1. Fork__ the ``cs-ranking`` repository on GitHub.

__ https://help.github.com/en/github/getting-started-with-github/fork-a-repo

2. Clone your fork locally:

   .. code-block:: bash

       $ git clone git@github.com:your_github_username_here/cs-ranking.git

3. Install your local copy using `poetry`_. This is how you set up your fork
   for local development::

   .. code-block:: bash

       $ cd cs-ranking
       $ poetry install


4. Set up `pre-commit`__ for this repository, which was installed in the
   previous step. This will make sure that the changes you make adhere to our
   coding standards by checking formatting with `black`.

__ https://pre-commit.com/

   .. code-block:: bash

       $ poetry run pre-commit install

   You can run all pre-commit hooks on all files using

   .. code-block:: bash

       $ poetry run pre-commit run --all-files

   The hooks will be run automatically when you run `git commit`.

5. Create a branch for your modifications:

   .. code-block:: bash

       $ git checkout -b some-name-for-the-branch

   Now you can make your changes locally.

6. When you're done making changes, check that the test suite still passes:

   .. code-block:: bash

       $ poetry run pytest

   Fetch some coffee.
   This might take several minutes.

7. Commit your changes and push your branch to GitHub:

   .. code-block:: bash

       $ git add .
       $ git commit
       $ git push origin some-name-for-the-branch

   Note that the second command will open an editor window in which you can
   write a commit message. Take care to use a `good, descriptive commit
   message`__. If one of the `pre-commit` hooks fails, read the error output
   and make the necessary fixes. The fixes may already have been made before
   you (as is the case with `black` formatting), you just need to `git add`
   them.

   If you do not understand why the hook fails, feel free to skip the hook for
   now:

   .. code-block:: bash

       $ git commit --no-verify

   And mention the problem in the pull request. We will happily help you figure
   out the source of the problem.

__ https://chris.beams.io/posts/git-commit/

8. Submit__ a pull request through the GitHub website.
   Keep the guidelines in the next section in mind.

__ https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request

Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~~~~

Before you submit a pull request, check that it meets these guidelines:

1. If you're adding new functionality, you should also include *tests* and
   *documentation* for that functionality. Put your new functionality into
   a function with a docstring, and add the feature to the list in README.rst.
   You can also add tests to the docstring:

   .. code-block:: python

       def my_awesome_new_fn(input_number):
           """A function that does something and returns something.

           Some extended documentation.

           Python code prefixed by `>>>` within the documentation doubles as
           a test case:

           >>> print("Hello, world!")
           Hello, world!
           >>> my_awesome_fn(42)
           43
           """
           return input_number + 1

3. After submitting the pull request, keep an eye on travis_ and make sure that
   the tests pass for all supported Python versions.

.. _travis: https://travis-ci.org/github/kiudee/cs-ranking/pull_requests

Tips
~~~~

To run a subset of tests:

.. code-block:: bash

    $ poetry run pytest <path-to-file>

Help Wanted
~~~~~~~~~~~

Look through the GitHub issues. Anything tagged with `"bug"`__ and `"help
wanted"`__ are particularly good places to get started. If you prefer to
implement new features, the `"enhancement"`__ tag might be interesting as well.

__ https://github.com/kiudee/cs-ranking/issues?q=is%3Aissue+is%3Aopen+label%3Abug
__ https://github.com/kiudee/cs-ranking/issues?q=is%3Aissue+is%3Aopen+label%3A%22help%20wanted%22
__ https://github.com/kiudee/cs-ranking/issues?q=is%3Aissue+is%3Aopen+label%3Aenhancement

Do Maintenance
==============

These tasks are mostly done by project maintainers, though if you think they
need to be done you can of course open an issue and ask for it. A pull request
is even better.

Deploying
~~~~~~~~~

Make sure all your changes are committed (including an entry in HISTORY.rst).
Then click on `"Draft a new release"`__ at the GitHub repository.
GitHub Actions will ensure that the package is uploaded to PyPI.

__ https://github.com/kiudee/cs-ranking/releases/new

.. _poetry: https://python-poetry.org/
