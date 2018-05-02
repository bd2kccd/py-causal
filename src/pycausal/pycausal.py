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
Updated on May 1, 2018

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

class pycausal():

    def start_vm(self, java_max_heap_size = None):
        tetrad_libdir = os.path.join(os.path.dirname(__file__), 'lib')

        for l in glob.glob(tetrad_libdir + os.sep + "*.jar"):
            javabridge.JARS.append(str(l))

        javabridge.start_vm(run_headless=True, max_heap_size = java_max_heap_size)
        javabridge.attach()        

    def stop_vm(self):
        javabridge.detach()
        javabridge.kill_vm()

    def isNodeExisting(self, nodes,node):
        try:
            nodes.index(node)
            return True
        except IndexError:
            print("Node {0} does not exist!".format(node))
            return False

    def loadMixedData(self, df, numCategoriesToDiscretize = 4):
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
                    value = javabridge.JClassWrapper('java.lang.Double')(df.iloc[row,col])
                    mixedDataBox.set(row,col,value)

                for col in disc_list:
                    cat_array = sorted(set(df[df.columns[col]]))
                    value = javabridge.JClassWrapper('java.lang.Integer')(cat_array.index(df.iloc[row,col]))
                    mixedDataBox.set(row,col,value)

            tetradData = javabridge.JClassWrapper('edu.cmu.tetrad.data.BoxDataSet')(mixedDataBox, node_list)

        else:
            # Generate random name
            temp_data_file = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(10)) + '.csv'
            temp_data_path = os.path.join(tempfile.gettempdir(), temp_data_file)
            df.to_csv(temp_data_path, sep = "\t", index = False)

            # Read Data from File
            f = javabridge.JClassWrapper('java.io.File')(temp_data_path)
            delimiter = javabridge.get_static_field('edu/pitt/dbmi/data/Delimiter','TAB','Ledu/pitt/dbmi/data/Delimiter;')
            dataReader = javabridge.JClassWrapper('edu.pitt.dbmi.data.reader.tabular.MixedTabularDataFileReader')(numCategoriesToDiscretize, f,delimiter)
            tetradData = dataReader.readInData()
            tetradData = javabridge.static_call('edu/pitt/dbmi/causal/cmd/util/TetradDataUtils','toDataModel','(Ledu/pitt/dbmi/data/Dataset;)Ledu/cmu/tetrad/data/DataModel;', tetradData)

            os.remove(temp_data_path)

        return tetradData

    def loadContinuousData(self, df, outputDataset = False):
        tetradData = None

        if(len(df.index)*df.columns.size <= 1500):

            dataBox = javabridge.JClassWrapper('edu.cmu.tetrad.data.DoubleDataBox')(len(df.index),df.columns.size)

            node_list = javabridge.JClassWrapper('java.util.ArrayList')()
            col_no = 0
            for col in df.columns:
                nodi = javabridge.JClassWrapper('edu.cmu.tetrad.data.ContinuousVariable')(col)
                node_list.add(nodi)

                for row in df.index:
                    value = javabridge.JClassWrapper('java.lang.Double')(df.iloc[row,col_no])
                    dataBox.set(row,col_no,value)

                col_no = col_no + 1

            tetradData = javabridge.JClassWrapper('edu.cmu.tetrad.data.BoxDataSet')(dataBox, node_list)

        else:
            #Generate random name
            temp_data_file = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(10)) + '.csv'
            temp_data_path = os.path.join(tempfile.gettempdir(), temp_data_file)
            df.to_csv(temp_data_path, sep = '\t', index = False)

            # Read Data from File
            f = javabridge.JClassWrapper('java.io.File')(temp_data_path)
            delimiter = javabridge.get_static_field('edu/pitt/dbmi/data/Delimiter','TAB','Ledu/pitt/dbmi/data/Delimiter;')
            dataReader = javabridge.JClassWrapper('edu.pitt.dbmi.data.reader.tabular.ContinuousTabularDataFileReader')(f,delimiter)
            tetradData = dataReader.readInData()
            tetradData = javabridge.static_call('edu/pitt/dbmi/causal/cmd/util/TetradDataUtils','toDataModel','(Ledu/pitt/dbmi/data/Dataset;)Ledu/cmu/tetrad/data/DataModel;', tetradData)

            os.remove(temp_data_path)

        if(not outputDataset):
            tetradData = javabridge.JClassWrapper('edu.cmu.tetrad.data.CovarianceMatrixOnTheFly')(tetradData)

        return tetradData

    def loadDiscreteData(self, df):
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
                    value = javabridge.JClassWrapper('java.lang.Integer')(cat_array.index(df.iloc[row,col_no]))
                    dataBox.set(row,col_no,value)

                col_no = col_no + 1

            tetradData = javabridge.JClassWrapper('edu.cmu.tetrad.data.BoxDataSet')(dataBox, node_list)

        else:
            # Generate random name
            temp_data_file = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(10)) + '.csv'
            temp_data_path = os.path.join(tempfile.gettempdir(), temp_data_file)
            df.to_csv(temp_data_path, sep = "\t", index = False)

            # Read Data from File
            f = javabridge.JClassWrapper('java.io.File')(temp_data_path)
            delimiter = javabridge.get_static_field('edu/pitt/dbmi/data/Delimiter','TAB','Ledu/pitt/dbmi/data/Delimiter;')
            dataReader = javabridge.JClassWrapper('edu.pitt.dbmi.data.reader.tabular.VerticalDiscreteTabularDataReader')(f,delimiter)
            tetradData = dataReader.readInData()
            tetradData = javabridge.static_call('edu/pitt/dbmi/causal/cmd/util/TetradDataUtils','toDataModel','(Ledu/pitt/dbmi/data/Dataset;)Ledu/cmu/tetrad/data/DataModel;', tetradData)

            os.remove(temp_data_path)

        return tetradData

    def restoreOriginalName(self, new_columns,orig_columns,node):
        if node[0] != 'L':
            index = new_columns.index(node)
            node = orig_columns[index]
        return node

    def extractTetradGraphNodes(self, tetradGraph, orig_columns = None, new_columns = None):
        n = tetradGraph.getNodes().toString()
        n = n[1:len(n)-1]
        n = n.split(",")
        for i in range(0,len(n)):
            n[i] = n[i].strip()
            if(orig_columns != None and new_columns != None):
                n[i] = pycausal.restoreOriginalName(self,new_columns,orig_columns,n[i])

        return n

    def extractTetradGraphEdges(self, tetradGraph, orig_columns = None, new_columns = None):
        e = tetradGraph.getEdges().toString()
        e = e[1:len(e)-1]
        e = e.split(",")    
        for i in range(0,len(e)):
            e[i] = e[i].strip()
            if(orig_columns != None and new_columns != None):
                token = e[i].split(" ")
                src = token[0]
                arc = token[1]
                dst = token[2]
                src = pycausal.restoreOriginalName(self,new_columns,orig_columns,src)
                dst = pycausal.restoreOriginalName(self,new_columns,orig_columns,dst)
                e[i] = src + " " + arc + " " + dst

        return e            

    def generatePyDotGraph(self, n,e,tetradGraph):
        graph = pydot.Dot(graph_type='digraph')

        # causal search and get edges
        tetradString = tetradGraph.toString()
        graph_edges = []
        token = tetradString.split('\n')
        for edge in token[4:-1]:
            if len(str(edge).split('. ')) > 1:
                graph_edges.append(str(edge).split('. ')[1])

        # gets the nodes in sorted order
        nodes_sorted = str(token[1]).split(',')
        nodes_sorted.sort()
        #for node in nodes_sorted:
        #    graph.add_node(pydot.Node(node))

        # create dictionaries of the nodes and edges
        nodes = {}
        edges = {}
        bootstraps = {}
        for edge in graph_edges:
            token = str(edge).split()
            n1 = token[0]
            arc = token[1]
            n2 = token[2]
            if n1 not in nodes: nodes[n1] = []
            if n2 not in nodes: nodes[n2] = []
            nodes[n1].append(n2)
            nodes[n2].append(n1)
            edges[n1, n2] = n1 + ' ' + arc + ' ' + n2
            if len(str(edge)) > 100:
                bootstraps[n1, n2] = str(edge[-100:])

        # graph plot the variables and edges
        for v0 in nodes.keys():
            for v1 in nodes.keys():
                if (v0, v1) in edges.keys():
                    arc = edges[v0, v1].split()[1]
                    if len(arc) >= 3:
                        edge = pydot.Edge(v0, v1)

                        if(arc[0] != "-"):
                            edge.set_dir("both")

                        if(arc[0] == "o"):
                            edge.set_arrowtail("odot")
                        elif(arc[0] == "<"):
                            edge.set_arrowtail("normal")

                        if(arc[2] == "-"):
                            edge.set_arrowhead("none")
                        elif(arc[2] == "o"):
                            edge.set_arrowhead("odot")
                        else:
                            edge.set_arrowhead("normal")

                        if len(bootstraps) > 0:
                            # nodes reported in sorted order
                            if nodes_sorted.index(v0) < nodes_sorted.index(v1): 
                                label = v0 + ' - ' + v1 + '\n' 
                            else:
                                label = v1 + ' - ' + v0 + '\n'            

                            # Bootstrapping distribution
                            # [no edge]
                            if '0.0000' not in bootstraps[v0, v1][0:16]:
                                label += bootstraps[v0, v1][0:16] + '\n'
                            for i in range(0,7):
                                e = bootstraps[v0, v1][16+i*12:28+i*12]
                                if '0.0000' not in e:                    
                                    label += e + '\n'

                            edge.set('fontname', 'courier')
                            edge.set('label', label)

                        graph.add_edge(edge)      

        return graph
