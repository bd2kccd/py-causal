'''

Copyright (C) 2015 University of Pittsburgh.
 
This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.
 
This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.
 
You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
MA 02110-1301  USA
 
Created on Feb 15, 2016

@author: Chirayu Wongchokprasitti, PhD 
@email: chw20@pitt.edu
'''

# lgpl 2.1
__author__ = 'Chirayu Kong Wongchokprasitti'
__version__ = '0.1.1'
__license__ = 'LGPL >= 2.1'

import javabridge
import os
import glob
import pydot
import random
import string
import tempfile

def start_vm(java_max_heap_size = None):
    tetrad_libdir = os.path.join(os.path.dirname(__file__), 'lib')

    for l in glob.glob(tetrad_libdir + os.sep + "*.jar"):
        javabridge.JARS.append(str(l))
            
    javabridge.start_vm(run_headless=True, max_heap_size = java_max_heap_size)
    javabridge.attach()        
    
def stop_vm():
    javabridge.detach()
    javabridge.kill_vm()

def isNodeExisting(nodes,node):
    try:
        nodes.index(node)
        return True
    except IndexError:
        print "Node %s does not exist!", node
        return False

def loadMixedData(df, numCategoriesToDiscretize = 4):
    tetradData = None

    if(len(df.index)*df.columns.size <= 1500):
        
        node_list = javabridge.JClassWrapper('java.util.ArrayList')()
        cont_list = []
        disc_list = []
        col_no = 0
        for col in df.columns:
            
            cat_array = sorted(set(df[col]))
            if(len(cat_array) > numCategoriesToDiscretize):
                # Continuous variable
                nodi = javabridge.JClassWrapper('edu.cmu.tetrad.data.ContinuousVariable')(col)
                node_list.add(nodi)
                
                cont_list.append(col_no)
            
            else:
                # Discrete variable
                cat_list = javabridge.JClassWrapper('java.util.ArrayList')()
                for cat in cat_array:
                    cat = str(cat)
                    cat_list.add(cat)
            
                nodname = javabridge.JClassWrapper('java.lang.String')(col)
                nodi = javabridge.JClassWrapper('edu.cmu.tetrad.data.DiscreteVariable')(nodname,cat_list)
                node_list.add(nodi)

                disc_list.append(col_no)
            
            col_no = col_no + 1
        
        mixedDataBox = javabridge.JClassWrapper('edu.cmu.tetrad.data.MixedDataBox')(node_list, len(df.index))
                                                                                    
        for row in df.index:

            for col in cont_list:
                value = javabridge.JClassWrapper('java.lang.Double')(df.ix[row][col])
                mixedDataBox.set(row,col,value)
            
            for col in disc_list:
                cat_array = sorted(set(df[df.columns[col]]))
                value = javabridge.JClassWrapper('java.lang.Integer')(cat_array.index(df.ix[row][col]))
                mixedDataBox.set(row,col,value)

        tetradData = javabridge.JClassWrapper('edu.cmu.tetrad.data.BoxDataSet')(mixedDataBox, node_list)
                    
    else:
        # Generate random name
        temp_data_file = ''.join(random.choice(string.lowercase) for i in range(10)) + '.csv'
        temp_data_path = os.path.join(tempfile.gettempdir(), temp_data_file)
        df.to_csv(temp_data_path, sep = "\t", index = False)
        
        # Read Data from File
        f = javabridge.JClassWrapper('java.io.File')(temp_data_path)
        delimiter = javabridge.get_static_field('edu/pitt/dbmi/data/Delimiter','TAB','Ledu/pitt/dbmi/data/Delimiter;')
        dataReader = javabridge.JClassWrapper('edu.pitt.dbmi.data.reader.tabular.MixedTabularDataFileReader')(numCategoriesToDiscretize, f,delimiter)
        tetradData = dataReader.readInDataFromFile(None)
        
        os.remove(temp_data_path)

    return tetradData

def loadContinuousData(df, outputDataset = False):
    tetradData = None
          
    if(len(df.index)*df.columns.size <= 1500):

        dataBox = javabridge.JClassWrapper('edu.cmu.tetrad.data.DoubleDataBox')(len(df.index),df.columns.size)

        node_list = javabridge.JClassWrapper('java.util.ArrayList')()
        col_no = 0
        for col in df.columns:
            nodi = javabridge.JClassWrapper('edu.cmu.tetrad.data.ContinuousVariable')(col)
            node_list.add(nodi)

            for row in df.index:
                value = javabridge.JClassWrapper('java.lang.Double')(df.ix[row][col_no])
                dataBox.set(row,col_no,value)
    
            col_no = col_no + 1

        tetradData = javabridge.JClassWrapper('edu.cmu.tetrad.data.BoxDataSet')(dataBox, node_list)

    else:
        #Generate random name
        temp_data_file = ''.join(random.choice(string.lowercase) for i in range(10)) + '.csv'
        temp_data_path = os.path.join(tempfile.gettempdir(), temp_data_file)
        df.to_csv(temp_data_path, sep = '\t', index = False)

        excludeVar = javabridge.JClassWrapper('java.util.HashSet')()
        excludeVar.add('MULT')

        # Read Data from File
        f = javabridge.JClassWrapper('java.nio.file.Paths').get(temp_data_path)
        dataReader = javabridge.JClassWrapper('edu.cmu.tetrad.io.TabularContinuousDataReader')(f,'\t')
        tetradData = dataReader.readInData(excludeVar)

        os.remove(temp_data_path)
    
    if(not outputDataset):
        tetradData = javabridge.JClassWrapper('edu.cmu.tetrad.data.CovarianceMatrixOnTheFly')(tetradData)

    return tetradData

def loadDiscreteData(df):
    tetradData = None
          
    if(len(df.index)*df.columns.size <= 1500):

        dataBox = javabridge.JClassWrapper('edu.cmu.tetrad.data.VerticalIntDataBox')(len(df.index),df.columns.size)

        node_list = javabridge.JClassWrapper('java.util.ArrayList')()
        col_no = 0
        for col in df.columns:

            cat_array = sorted(set(df[col]))
            cat_list = javabridge.JClassWrapper('java.util.ArrayList')()
            for cat in cat_array:
                cat = str(cat)
                cat_list.add(cat)

            nodname = javabridge.JClassWrapper('java.lang.String')(col)
            nodi = javabridge.JClassWrapper('edu.cmu.tetrad.data.DiscreteVariable')(nodname,cat_list)
            node_list.add(nodi)

            for row in df.index:
                value = javabridge.JClassWrapper('java.lang.Integer')(cat_array.index(df.ix[row][col_no]))
                dataBox.set(row,col_no,value)

            col_no = col_no + 1

        tetradData = javabridge.JClassWrapper('edu.cmu.tetrad.data.BoxDataSet')(dataBox, node_list)

    else:
        # Generate random name
        temp_data_file = ''.join(random.choice(string.lowercase) for i in range(10)) + '.csv'
        temp_data_path = os.path.join(tempfile.gettempdir(), temp_data_file)
        df.to_csv(temp_data_path, sep = "\t", index = False)

        excludeVar = javabridge.JClassWrapper('java.util.HashSet')()
        excludeVar.add("MULT")

        # Read Data from File
        f = javabridge.JClassWrapper('java.nio.file.Paths').get(temp_data_path)
        dataReader = javabridge.JClassWrapper('edu.cmu.tetrad.io.VerticalTabularDiscreteDataReader')(f,'\t')
        tetradData = dataReader.readInData(excludeVar)

        os.remove(temp_data_path)
        
    return tetradData

def extractTetradGraphNodes(tetradGraph):
    n = tetradGraph.getNodes().toString()
    n = n[1:len(n)-1]
    n = n.split(",")
    for i in range(0,len(n)):
        node = n[i]
        n[i] = node.strip()

    return n

def extractTetradGraphEdges(tetradGraph):
    e = tetradGraph.getEdges().toString()
    e = e[1:len(e)-1]
    e = e.split(",")    
    for i in range(0,len(e)):
        e[i] = e[i].strip()

    return e            
    
def generatePyDotGraph(n,e):
    graph = pydot.Dot(graph_type='digraph')
    nodes = []

    for i in range(0,len(n)):
        nodes.append(pydot.Node(n[i]))
        graph.add_node(nodes[i])

    for i in range(0,len(e)):
        token = e[i].split(" ")
        if(len(token) >= 3):
            src = token[0]
            arc = token[1]
            dst = token[2]
            if(isNodeExisting(n,src) and isNodeExisting(n,dst)):
                edge = pydot.Edge(nodes[n.index(src)],nodes[n.index(dst)])
                if(arc == "---"):
                    edge.set_arrowhead("none")
                graph.add_edge(edge)

    return graph
