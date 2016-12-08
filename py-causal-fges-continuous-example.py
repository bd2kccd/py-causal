#!/usr/local/bin/python


import os
import pandas as pd
import pydot
from IPython.display import SVG

data_dir = os.path.join(os.getcwd(), 'data', 'charity.txt')
df = pd.read_table(data_dir, sep="\t")

from pycausal import pycausal as pc

pc.start_vm()

from pycausal import search as s

fges = s.fges(df,penaltydiscount = 2, maxDegree = -1,
            faithfulnessAssumed = True, numofthreads = 2, verbose = True)

fges.getNodes()
fges.getEdges()

dot = fges.getDot()
svg_str = dot.create_svg(prog='dot')

f = open('fges-continuous.dot','w')
f.write(svg_str)
f.close()

pc.stop_vm()
