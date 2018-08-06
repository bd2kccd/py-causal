#!/usr/local/bin/python


import os
import pandas as pd
import pydot
from IPython.display import SVG

data_dir = os.path.join(os.getcwd(), 'data', 'charity.txt')
df = pd.read_table(data_dir, sep="\t")

from pycausal.pycausal import pycausal as pc
pc = pc()
pc.start_vm()

from pycausal import search as s
tetrad = s.tetradrunner()
tetrad.run(algoId = 'fges', dfs = df, scoreId = 'sem-bic', dataType = 'continuous',
           penaltyDiscount = 2, maxDegree = -1, faithfulnessAssumed = True, verbose = True)

tetrad.getNodes()
tetrad.getEdges()

dot = tetrad.getDot()
dot.write_svg('fges-continuous.svg')

pc.stop_vm()
