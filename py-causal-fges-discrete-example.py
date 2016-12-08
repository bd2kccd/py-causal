#!/usr/local/bin/python


import os
import pandas as pd
import pydot
from IPython.display import SVG

data_dir = os.path.join(os.getcwd(), 'data', 'audiology.txt')
df = pd.read_table(data_dir, sep="\t")

from pycausal import pycausal as pc

pc.start_vm(java_max_heap_size = '100M')

from pycausal import search as s

fges = s.fgesDiscrete(df,structurePrior = 1.0, samplePrior = 1.0, 
                            maxDegree = 3, faithfulnessAssumed = True, numOfThreads = 2, 
                            verbose = True)

fges.getNodes()
fges.getEdges()

dot = fges.getDot()
svg_str = dot.create_svg(prog='dot')

f = open('fges-discrete.dot','w')
f.write(svg_str)
f.close()

pc.stop_vm()
