#!/usr/local/bin/python


import os
import pandas as pd
import pydot
from IPython.display import SVG

data_dir = os.path.join(os.getcwd(), 'data', 'audiology.txt')
df = pd.read_table(data_dir, sep="\t")

from pycausal.pycausal import pycausal as pc
pc = pc()
pc.start_vm(java_max_heap_size = '100M')

from pycausal import search as s
tetrad = s.tetradrunner()
tetrad.run(algoId = 'fges', dfs = df, scoreId = 'bdeu', dataType = 'discrete',
           structurePrior = 1.0, samplePrior = 1.0, maxDegree = 3, faithfulnessAssumed = True, verbose = True)

tetrad.getNodes()
tetrad.getEdges()

dot_str = pc.tetradGraphToDot(tetrad.getTetradGraph())
graphs = pydot.graph_from_dot_data(dot_str)
graphs[0].write_svg('fges-discrete.svg')

pc.stop_vm()
