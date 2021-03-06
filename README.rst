=======
wsnsims
=======

This project provides simulations of several federation algorithms used in
wireless sensor networks (WSNs). At present, the implemented algorithms are,

* FLOWER
* TOCS
* FOCUS
* MINDS
* LOAF

Setup
=====

This project can be set up using standard Python tools. It relies on vanilla
Python 3.5+ as its interpreter. It is highly recommended that you install this
package in a virtual environment(VirtualEnv or Miniconda works great).::

    $ git clone https://<projecturl>/wsnsims.git
    $ pip install -e wsnsims

Running
=======

At present, running the simulations is an all-or-nothing affair. All results
will be placed in a local directory called ``results`` by default. From within
the ``wsnsims`` directory, execute the following.::

    $ python -m wsnsims.conductor.driver
    
    # LOAF only
    $ python -m wsnsims.conductor.loafdriver

For options, pass in the ``--help`` option::

    $ python -m wsnsims.conductor.driver --help
