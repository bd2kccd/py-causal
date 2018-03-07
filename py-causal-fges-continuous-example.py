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
tetrad.run(algoId = 'fges', dfs = df, scoreId = 'sem-bic', priorKnowledge = prior, dataType = 0,
           penaltyDiscount = 2, maxDegree = -1, faithfulnessAssumed = True, verbose = True)

tetrad.getNodes()
tetrad.getEdges()

dot = tetrad.getDot()
svg_str = dot.create_svg(prog='dot')

f = open('fges-continuous.dot','w')
f.write(svg_str)
f.close()

pc.stop_vm()
