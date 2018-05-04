#!/usr/local/bin/python


import os
import pandas as pd
import pydot
from IPython.display import SVG

data_dir = os.path.join(os.getcwd(), 'data', 'audiology.txt')
df = pd.read_table(data_dir, sep="\t")

from pycausal import pycausal as pc

pc.start_vm(java_max_heap_size = '100M')
pc = pc()
from pycausal import search as s
tetrad = s.tetradrunner()
tetrad.run(algoId = 'fges', dfs = df, scoreId = 'bdeu', priorKnowledge = prior, dataType = 1,
           structurePrior = 1.0, samplePrior = 1.0, maxDegree = 3, faithfulnessAssumed = True, verbose = True)

tetrad.getNodes()
tetrad.getEdges()

dot = tetrad.getDot()
svg_str = dot.create_svg(prog='dot')

f = open('fges-discrete.dot','w')
f.write(svg_str)
f.close()

pc.stop_vm()
