#!/usr/local/bin/python


import os
import pandas as pd
import pydot
from IPython.display import SVG


data_dir = os.path.join(os.getcwd(), 'data', 'charity.txt')
df = pd.read_table(data_dir, sep="\t")



from tetrad import search as s

fgs = s.fgs(df,penaltydiscount = 2, depth = -1,
            faithfulness = True, verbose = True, java_max_heap_size = '500M')



fgs.getNodes()
fgs.getEdges()


dot = fgs.getDot()
svg_str = dot.create_svg(prog='dot')

f = open('tetrad.dot','w')
f.write(svg_str)
f.close()

