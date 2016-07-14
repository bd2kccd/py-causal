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

fgs = s.fgsDiscrete(df,structurePrior = 1.0, samplePrior = 1.0, 
                            depth = 3, heuristicSpeedup = True, numOfThreads = 2, 
                            verbose = True)

fgs.getNodes()
fgs.getEdges()

dot = fgs.getDot()
svg_str = dot.create_svg(prog='dot')

f = open('fgs-discrete.dot','w')
f.write(svg_str)
f.close()

pc.stop_vm()
