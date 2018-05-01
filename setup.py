#!/usr/bin/env python
'''

Copyright (C) 2015 University of Pittsburgh.
 
This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.
 
This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.
 
You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
MA 02110-1301  USA
 
Created on Feb 15, 2016
Updated on May 1, 2018

@author: chw20
'''

from distutils.core import setup

setup(
    name = "pycausal",
    version = "1.1.0",
    description = "Python wrapper for the Tetrad Library",
    author = "Chirayu Kong Wongchokprasitti",
    author_email = 'chw20@pitt.edu',
    url = 'http://github.com/bd2kccd/py-causal',
    download_url = 'https://github.com/bd2kccd/py-causal/archive/v1.0.0.tar.gz',
    license = 'Lesser GNU General Public License version >= 2.1 (LGPL >= 2.1)',
    keywords = 'tetrad causal graph bayesian network search discovery',
    platforms = ['any'],
    classifiers =    ['Development Status :: 5 - Beta',
        'Intended Audience :: Science/Research',
        'License :: LGPL 2.1 License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules'],
    package_dir = {'': 'src'},
    packages = ['pycausal'],
    package_data = {
        'pycausal': ['lib/*.jar'],
    },
    install_requires = [
        'javabridge>=1.0.11', 
        'pydot',
        'pyparsing',
        'GraphViz'],
    data_files = [('.', ['LICENSE', 'README'], 'data/charity.txt','data/audiology.txt')]
)

