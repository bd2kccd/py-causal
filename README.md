pytetrad
========
Python APIs for causal modeling algorithms developed by the University of Pittsburgh/Carnegie Mellon University [Center for Causal Discovery](http://www.ccd.pitt.edu). 


This code is distributed under the LGPL 2.1 license.

Requirements:
============

Python 2.7 (does not work with Python 3)
*javabridge>=1.0.11
*pandas
*numpy 
*pydot
*GraphViz
* JDK 1.7+

Installation overview:
======================
We have found two approaches to be useful:
* Direct python installation with pip, possibly including use of [Jupyter](http://jupyter.org/). This approach is likely best for users who have Python installed and are familiar with installing Python modules.
* Installation via [Anaconda](https://www.continuum.io/downloads), which  installs Python and related utilities.

Directions for both approache are given below...

Installation with pip
=====================

If you do not have pip installed already, try [these instructions](https://pip.pypa.io/en/stable/installing/).

Once pip is installed, execute these commands

* pip install javabridge
* pip install pandas
* pip install numpy
* pip install pydot
* pip install GraphViz


After running this command, enter a python shell and attempt the follwing imports
 * import pandas as pd
 * import pydot
 * from tetrad import search as s
 
We have observed that on some OS X installations, pydot may provide the following response
    Couldn't import dot_parser, loading of dot files will not be possible.

If you see this, try the following
uninstall pydot
pip install pyparsing==1.5.7
install pydot


Then, from within the pytetrad directory, run the following command:

    python setup.py install

Finally, try to run the python example

    python pytetrad-example.py

Be sure to run this from within the pytetrad directory.

This program will create a file named tetrad.svg, which should be viewable in any SVG capable program. If you see a causal graph, everything is working correctly.

Running Jupyter/IPython
-----------------------

We have found [Jupyter](http://jupyter.org/) notebooks to be helpful. (Those who have run IPython in the past should know that Jupyter is simply a new name for IPython). To add Jupyter to your completed python install, simply run

    pip -U jupyter
    jupyter notebook
 
 
 and then load one of the Jupyter notebooks found in this installation. 

Anaconda/Jupyter
================

Installing Python with Anaconda and Jupyter may be easier for some users:

* [Download and install Anaconda](https://www.continuum.io/downloads)
* conda install python-javabridge

For OS X, this default install does not seem to work well. try the following instead:
        conda install --channel https://conda.anaconda.org/david_baddeley python-javabridge

* conda install pandas  
* conda install numpy
* conda install pydot
* conda install graphviz 
* conda install -c https://conda.anaconda.org/chirayu pytetrad 
* jupyter notebook

and the load one of the Jupyter notebooks.

