'''
Created on Feb 17, 2016

@author: chw20
'''

import javabridge
import os
import glob
import pydot
import random
import string
import tempfile

import tetrad

class fgs():
    
    graph = None
    nodes = []
    edges = []
    
    def __init__(self, df, penaltydiscount = 4, depth = 3, faithfulness = True, verbose = False, java_max_heap_size = None):
    
        tetrad_libdir = os.path.join(os.path.dirname(__file__), 'lib')
        # print 'tetrad_libdir: %s' % tetrad_libdir
        
        for l in glob.glob(tetrad_libdir + os.sep + "*.jar"):
            javabridge.JARS.append(str(l))
            
        javabridge.start_vm(run_headless=True, max_heap_size = java_max_heap_size)
        javabridge.attach()
            
        tetradData = None
          
        if(len(df.index)*df.columns.size <= 1500):

            node_list = javabridge.JWrapper(javabridge.make_instance("java/util/ArrayList", "()V"))
            for col in df.columns:
                nodname = javabridge.make_instance("java/lang/String", "(Ljava/lang/String;)V",col)
                nodi = javabridge.make_instance("edu/cmu/tetrad/graph/GraphNode", "(Ljava/lang/String;)V",nodname)
                node_list.add(nodi)
                
            tetradMatrix = javabridge.JWrapper(javabridge.make_instance("edu/cmu/tetrad/util/TetradMatrix","(II)V",
                                            len(df.index),df.columns.size))
        
            for row in df.index:
                for col in range(0,df.columns.size):
                    tetradMatrix.set(row,col,df.ix[row][col])            
            
            tetradData = javabridge.static_call("edu/cmu/tetrad/data/ColtDataSet","makeContinuousData",
                            "(Ljava/util/List;Ledu/cmu/tetrad/util/TetradMatrix;)Ledu/cmu/tetrad/data/DataSet;",
                            node_list,tetradMatrix)

        else:
            #Generate random name
            temp_data_file = ''.join(random.choice(string.lowercase) for i in range(10)) + '.csv'
            temp_data_path = os.path.join(tempfile.gettempdir(), temp_data_file)
            df.to_csv(temp_data_path, sep = "\t")

            f = javabridge.make_instance("java/io/File", "(Ljava/lang/String;)V",temp_data_path)
            excludeVar = javabridge.JWrapper(javabridge.make_instance("java/util/HashSet","()V"))
            excludeVar.add("MULT")
            tetradData = javabridge.static_call("edu/cmu/tetrad/data/BigDataSetUtility","readInContinuousData","(Ljava/io/File;CLjava/util/Set;)Ledu/cmu/tetrad/data/DataSet;",f,"\t",excludeVar)
            os.remove(temp_data_path)
            
        fgs = javabridge.make_instance("edu/cmu/tetrad/search/Fgs","(Ledu/cmu/tetrad/data/DataSet;)V",tetradData)
        fgs = javabridge.JWrapper(fgs)
    
        fgs.setPenaltyDiscount(penaltydiscount)# set to 2 if variable# <= 50 otherwise set it to 4
        fgs.setDepth(depth)#-1
        fgs.setNumPatternsToStore(0)
        fgs.setFaithfulnessAssumed(faithfulness)
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
                if(tetrad.isNodeExisting(n,src) and tetrad.isNodeExisting(n,dst)):
                    edge = pydot.Edge(nodes[n.index(src)],nodes[n.index(dst)])
                    if(arc == "---"):
                        edge.set_arrowhead("none")
                    graph.add_edge(edge)
    
        self.edges = e            
    
        javabridge.detach()
        javabridge.kill_vm()
            
        self.graph = graph
        
    def getDot(self):
        return self.graph
    
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):    
        return self.edges

