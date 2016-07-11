'''
Created on Feb 17, 2016

@author: Chirayu Wongchokprasitti, PhD 
@email: chw20@pitt.edu
'''

import javabridge
import os
import glob
import pydot
import random
import string
import tempfile

import pycausal

class fgsDiscrete():
    
    graph = None
    nodes = []
    edges = []
    
    def __init__(self, df, structurePrior = 1.0, samplePrior = 1.0, depth = 3, heuristicSpeedup = True, numofthreads = 2, verbose = False, java_max_heap_size = None, priorKnowledge = None, stop_jvm_after_finish = True):
            
        tetrad_libdir = os.path.join(os.path.dirname(__file__), 'lib')
        
        for l in glob.glob(tetrad_libdir + os.sep + "*.jar"):
            javabridge.JARS.append(str(l))
            
        javabridge.start_vm(run_headless=True, max_heap_size = java_max_heap_size)
        javabridge.attach()
            
        score = None
          
        if(len(df.index)*df.columns.size <= 1500):
        
            dataBox = javabridge.JClassWrapper('edu.cmu.tetrad.data.VerticalIntDataBox')(len(df.index),df.columns.size)

            node_list = javabridge.JClassWrapper('java.util.ArrayList')()
            col_no = 0
            for col in df.columns:

                cat_array = sorted(set(df[col]))
                cat_list = javabridge.JClassWrapper('java.util.ArrayList')()
                for cat in cat_array:
                    catname = javabridge.JClassWrapper('java.lang.String')(cat)
                    cat_list.add(catname)

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

            # Read in method 1
            f = javabridge.JClassWrapper('java.nio.file.Paths').get(temp_data_path)
            dataReader = javabridge.JClassWrapper('edu.cmu.tetrad.io.VerticalTabularDiscreteDataReader')(f,'\t')
            dataReader = javabridge.JWrapper(dataReader)
            tetradData = dataReader.readInData(excludeVar)
            
            # Read in method 2 -- Depreciated
            # f = javabridge.make_instance("java/io/File", "(Ljava/lang/String;)V",temp_data_path)
            # tetradData = javabridge.static_call("edu/cmu/tetrad/data/BigDataSetUtility", "readInDiscreteData", "(Ljava/io/File;CLjava/util/Set;)Ledu/cmu/tetrad/data/DataSet;",f,"\t",excludeVar)
            
            os.remove(temp_data_path)
        
        score = javabridge.JClassWrapper('edu.cmu.tetrad.search.BDeuScore')(tetradData)
        score.setStructurePrior(structurePrior)
        score.setSamplePrior(samplePrior)
        
        fgs = javabridge.JClassWrapper('edu.cmu.tetrad.search.Fgs')(score)
        fgs.setDepth(depth)
        fgs.setNumPatternsToStore(0)
        fgs.setHeuristicSpeedup(heuristicSpeedup)
        fgs.setParallelism(numofthreads)
        fgs.setVerbose(verbose)
        tetradGraph = fgs.search()
        
        graph = pydot.Dot(graph_type='digraph')
        
        n = tetradGraph.getNodeNames().toString()
        n = n[1:len(n)-1]
        n = n.split(",")
        
        nodes = []
        
        for i in range(0,len(n)):
            node = n[i]
            n[i] = node.strip()
            nodes.append(pydot.Node(n[i]))
            graph.add_node(nodes[i])
        
        self.nodes = n
        
        e = tetradGraph.getEdges().toString()
        e = e[1:len(e)-1]
        e = e.split(",")    
    
        for i in range(0,len(e)):
            e[i] = e[i].strip()
            token = e[i].split(" ")
            if(len(token) == 3):
                src = token[0]
                arc = token[1]
                dst = token[2]
                if(pycausal.isNodeExisting(n,src) and pycausal.isNodeExisting(n,dst)):
                    edge = pydot.Edge(nodes[n.index(src)],nodes[n.index(dst)])
                    if(arc == "---"):
                        edge.set_arrowhead("none")
                    graph.add_edge(edge)
    
        self.edges = e            
    
        javabridge.detach()
        if stop_jvm_after_finish == True:
            javabridge.kill_vm()
            
        self.graph = graph
        
    def getDot(self):
        return self.graph
    
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):    
        return self.edges

    
class fgs():
    
    graph = None
    nodes = []
    edges = []
    
    def __init__(self, df, penaltydiscount = 4, depth = 3, faithfulness = True, numofthreads = 2, verbose = False, java_max_heap_size = None, priorKnowledge = None, stop_jvm_after_finish = True):
    
        tetrad_libdir = os.path.join(os.path.dirname(__file__), 'lib')
        
        for l in glob.glob(tetrad_libdir + os.sep + "*.jar"):
            javabridge.JARS.append(str(l))
            
        javabridge.start_vm(run_headless=True, max_heap_size = java_max_heap_size)
        javabridge.attach()
            
        tetradData = None
          
        if(len(df.index)*df.columns.size <= 1500):

            node_list = javabridge.JClassWrapper('java.util.ArrayList')()
            for col in df.columns:
                nodi = javabridge.JClassWrapper('edu.cmu.tetrad.data.ContinuousVariable')(col)
                node_list.add(nodi)
            
            dataBox = javabridge.JClassWrapper('edu.cmu.tetrad.data.DoubleDataBox')(len(df.index),df.columns.size)
            
            for row in df.index:
                for col in range(0,df.columns.size):
                    value = javabridge.JClassWrapper('java.lang.Double')(df.ix[row][col])
                    dataBox.set(row,col,value)
                        
            tetradData = javabridge.JClassWrapper('edu.cmu.tetrad.data.BoxDataSet')(dataBox, node_list)
            
        else:
            #Generate random name
            temp_data_file = ''.join(random.choice(string.lowercase) for i in range(10)) + '.csv'
            temp_data_path = os.path.join(tempfile.gettempdir(), temp_data_file)
            df.to_csv(temp_data_path, sep = '\t', index = False)
            
            excludeVar = javabridge.JClassWrapper('java.util.HashSet')()
            excludeVar.add('MULT')

            # Read in method 1
            f = javabridge.JClassWrapper('java.nio.file.Paths').get(temp_data_path)
            dataReader = javabridge.JClassWrapper('edu.cmu.tetrad.io.TabularContinuousDataReader')(f,'\t')
            tetradData = dataReader.readInData(excludeVar)
            
            # Read in method 2 -- Depreciated
            # f = javabridge.JClassWrapper('java.io.File')(temp_data_path)
            # tetradData = javabridge.JClassWrapper('edu.cmu.tetrad.data.BigDataSetUtility').readInContinuousData(f,'\t',excludeVar)
            
            os.remove(temp_data_path)
        
        tetradData = javabridge.JClassWrapper('edu.cmu.tetrad.data.CovarianceMatrixOnTheFly')(tetradData)

        score = javabridge.JClassWrapper('edu.cmu.tetrad.search.SemBicScore')(tetradData)
        score.setPenaltyDiscount(penaltydiscount) # set to 2 if variable# <= 50 otherwise set it to 4
        
        fgs = javabridge.JClassWrapper('edu.cmu.tetrad.search.Fgs')(score)
        fgs.setDepth(depth)#-1
        fgs.setNumPatternsToStore(0)
        fgs.setFaithfulnessAssumed(faithfulness)
        fgs.setParallelism(numofthreads)
        fgs.setVerbose(verbose)
        tetradGraph = fgs.search()
        
        graph = pydot.Dot(graph_type='digraph')
        
        n = tetradGraph.getNodeNames().toString()
        n = n[1:len(n)-1]
        n = n.split(",")
        
        nodes = []
        
        for i in range(0,len(n)):
            node = n[i]
            n[i] = node.strip()
            nodes.append(pydot.Node(n[i]))
            graph.add_node(nodes[i])
        
        self.nodes = n
        
        e = tetradGraph.getEdges().toString()
        e = e[1:len(e)-1]
        e = e.split(",")    
    
        for i in range(0,len(e)):
            e[i] = e[i].strip()
            token = e[i].split(" ")
            if(len(token) == 3):
                src = token[0]
                arc = token[1]
                dst = token[2]
                if(pycausal.isNodeExisting(n,src) and pycausal.isNodeExisting(n,dst)):
                    edge = pydot.Edge(nodes[n.index(src)],nodes[n.index(dst)])
                    if(arc == "---"):
                        edge.set_arrowhead("none")
                    graph.add_edge(edge)
    
        self.edges = e            
    
        javabridge.detach()
        if stop_jvm_after_finish == True:
            javabridge.kill_vm()
            
        self.graph = graph
        
    def getDot(self):
        return self.graph
    
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):    
        return self.edges

