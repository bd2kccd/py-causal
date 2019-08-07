#!/usr/local/bin/python


import os
import pandas as pd
import pydot

data_dir = os.path.join(os.getcwd(), 'data', 'charity.txt')
df = pd.read_table(data_dir, sep="\t")

from pycausal.pycausal import pycausal as pc
pc = pc()
pc.start_vm()

from pycausal import search as s
tetrad = s.tetradrunner()
tetrad.run(algoId = 'fges', dfs = df, scoreId = 'sem-bic-score', dataType = 'continuous',
           maxDegree = -1, faithfulnessAssumed = True, verbose = True)

tetrad.getNodes()
tetrad.getEdges()

dot_str = pc.tetradGraphToDot(tetrad.getTetradGraph())
graphs = pydot.graph_from_dot_data(dot_str)
graphs[0].write_svg('fges-continuous.svg')

pc.stop_vm()
