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

fgs = s.fgs(df,penaltydiscount = 2, maxDegree = -1,
            faithfulnessAssumed = True, numofthreads = 2, verbose = True)

fgs.getNodes()
fgs.getEdges()

dot = fgs.getDot()
svg_str = dot.create_svg(prog='dot')

f = open('fgs-continuous.dot','w')
f.write(svg_str)
f.close()

pc.stop_vm()
