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
    
    def __init__(self, df, structurePrior = 1.0, samplePrior = 1.0, depth = 3, heuristicSpeedup = True, numofthreads = 2, verbose = False, priorKnowledge = None):
        
        tetradData = pycausal.loadDiscreteData(df)
        
        score = javabridge.JClassWrapper('edu.cmu.tetrad.search.BDeuScore')(tetradData)
        score.setStructurePrior(structurePrior)
        score.setSamplePrior(samplePrior)
        
        fgs = javabridge.JClassWrapper('edu.cmu.tetrad.search.Fgs')(score)
        fgs.setDepth(depth)
        fgs.setNumPatternsToStore(0)
        fgs.setHeuristicSpeedup(heuristicSpeedup)
        fgs.setParallelism(numofthreads)
        fgs.setVerbose(verbose)
        
        if priorKnowledge is not None:    
            fgs.setKnowledge(priorKnowledge)
            
        tetradGraph = fgs.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(tetradGraph)            
        self.graph = pycausal.generatePyDotGraph(self.nodes,self.edges)
        
        
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
    
    def __init__(self, df, penaltydiscount = 4, depth = 3, ignoreLinearDependence = True, heuristicSpeedup = True, numofthreads = 2, verbose = False, priorKnowledge = None):
            
        tetradData = pycausal.loadContinuousData(df)

        score = javabridge.JClassWrapper('edu.cmu.tetrad.search.SemBicScore')(tetradData)
        score.setPenaltyDiscount(penaltydiscount) # set to 2 if variable# <= 50 otherwise set it to 4
        
        fgs = javabridge.JClassWrapper('edu.cmu.tetrad.search.Fgs')(score)
        fgs.setDepth(depth)
        fgs.setNumPatternsToStore(0)
        fgs.setIgnoreLinearDependent(ignoreLinearDependence)
        fgs.setHeuristicSpeedup(heuristicSpeedup)
        fgs.setParallelism(numofthreads)
        fgs.setVerbose(verbose)
        
        if priorKnowledge is not None:    
            fgs.setKnowledge(priorKnowledge)
            
        tetradGraph = fgs.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(tetradGraph)            
        self.graph = pycausal.generatePyDotGraph(self.nodes,self.edges)
        
    def getDot(self):
        return self.graph
    
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):    
        return self.edges

class fci():
    
    nodes = []
    edges = []
    
    def __init__(self, df, continuous = True, depth = 3, significance = 0.05, noDSepSearch = False, verbose = False, priorKnowledge = None):
        IndTest = None
        
        if(continuous):
            tetradData = pycausal.loadContinuousData(df)
            IndTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, significance)
        else:
            tetradData = pycausal.loadDiscreteData(df)
            IndTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, significance)
        
        fci = javabridge.JClassWrapper('edu.cmu.tetrad.search.Fci')(IndTest)
        fci.setDepth(depth)
        fci.setPossibleDsepSearchDone(not noDSepSearch)
        fci.setVerbose(verbose)
        
        if priorKnowledge is not None:    
            fci.setKnowledge(priorKnowledge)
            
        tetradGraph = fci.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(tetradGraph) 
        
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):    
        return self.edges
        
class cfci():
    
    nodes = []
    edges = []
    
    def __init__(self, df, continuous = True, depth = 3, significance = 0.05, verbose = False, priorKnowledge = None):
        IndTest = None
        
        if(continuous):
            tetradData = pycausal.loadContinuousData(df)
            IndTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, significance)
        else:
            tetradData = pycausal.loadDiscreteData(df)
            IndTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, significance)
        
        cfci = javabridge.JClassWrapper('edu.cmu.tetrad.search.Cfci')(IndTest)
        cfci.setDepth(depth)
        cfci.setVerbose(verbose)
        
        if priorKnowledge is not None:    
            cfci.setKnowledge(priorKnowledge)
            
        tetradGraph = cfci.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(tetradGraph) 
        
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):    
        return self.edges
        
class ccd():
    
    nodes = []
    edges = []
    
    def __init__(self, df, continuous = True, depth = 3, significance = 0.05, verbose = False, priorKnowledge = None):
        IndTest = None
        
        if(continuous):
            tetradData = pycausal.loadContinuousData(df)
            IndTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, significance)
        else:
            tetradData = pycausal.loadDiscreteData(df)
            IndTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, significance)
        
        ccd = javabridge.JClassWrapper('edu.cmu.tetrad.search.Ccd')(IndTest)
        ccd.setDepth(depth)
        ccd.setVerbose(verbose)
        
        if priorKnowledge is not None:    
            ccd.setKnowledge(priorKnowledge)
            
        tetradGraph = ccd.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(tetradGraph) 
        
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):    
        return self.edges
    
class pc():
    
    graph = None
    nodes = []
    edges = []
    
    def __init__(self, df, continuous = True, depth = 3, aggressivelyPreventCycles = False, falseDiscoveryRate = False, significance = 0.05, verbose = False, priorKnowledge = None):
        IndTest = None
    
        if(continuous):
            tetradData = pycausal.loadContinuousData(df)
            IndTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, significance)
        else:
            tetradData = pycausal.loadDiscreteData(df)
            IndTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, significance)
        
        pc = javabridge.JClassWrapper('edu.cmu.tetrad.search.Pc')(IndTest)
        pc.setDepth(depth)
        pc.setAggressivelyPreventCycles(aggressivelyPreventCycles)
        pc.setFdr(falseDiscoveryRate)
        pc.setVerbose(verbose)
        
        if priorKnowledge is not None:    
            pc.setKnowledge(priorKnowledge)
            
        tetradGraph = pc.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(tetradGraph)
        self.graph = pycausal.generatePyDotGraph(self.nodes,self.edges)

    def getDot(self):
        return self.graph
        
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):    
        return self.edges
    
class pcstable():
    
    graph = None
    nodes = []
    edges = []
    
    def __init__(self, df, continuous = True, depth = 3, aggressivelyPreventCycles = False, significance = 0.05, verbose = False, priorKnowledge = None):
        IndTest = None
    
        if(continuous):
            tetradData = pycausal.loadContinuousData(df)
            IndTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, significance)
        else:
            tetradData = pycausal.loadDiscreteData(df)
            IndTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, significance)
        
        pcstable = javabridge.JClassWrapper('edu.cmu.tetrad.search.PcStable')(IndTest)
        pcstable.setDepth(depth)
        pcstable.setAggressivelyPreventCycles(aggressivelyPreventCycles)
        pcstable.setVerbose(verbose)
        
        if priorKnowledge is not None:    
            pcstable.setKnowledge(priorKnowledge)
            
        tetradGraph = pcstable.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(tetradGraph)
        self.graph = pycausal.generatePyDotGraph(self.nodes,self.edges)

    def getDot(self):
        return self.graph
        
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):    
        return self.edges    

class cpc():
    
    graph = None
    nodes = []
    edges = []
    
    def __init__(self, df, continuous = True, depth = 3, aggressivelyPreventCycles = False, significance = 0.05, verbose = False, priorKnowledge = None):
        IndTest = None
    
        if(continuous):
            tetradData = pycausal.loadContinuousData(df)
            IndTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, significance)
        else:
            tetradData = pycausal.loadDiscreteData(df)
            IndTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, significance)
        
        cpc = javabridge.JClassWrapper('edu.cmu.tetrad.search.Cpc')(IndTest)
        cpc.setDepth(depth)
        cpc.setAggressivelyPreventCycles(aggressivelyPreventCycles)
        cpc.setVerbose(verbose)
        
        if priorKnowledge is not None:    
            cpc.setKnowledge(priorKnowledge)
            
        tetradGraph = cpc.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(tetradGraph)
        self.graph = pycausal.generatePyDotGraph(self.nodes,self.edges)

    def getDot(self):
        return self.graph
        
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):    
        return self.edges    
    
class cpcstable():
    
    graph = None
    nodes = []
    edges = []
    
    def __init__(self, df, continuous = True, depth = 3, aggressivelyPreventCycles = False, significance = 0.05, verbose = False, priorKnowledge = None):
        IndTest = None
    
        if(continuous):
            tetradData = pycausal.loadContinuousData(df)
            IndTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, significance)
        else:
            tetradData = pycausal.loadDiscreteData(df)
            IndTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, significance)
        
        cpcstable = javabridge.JClassWrapper('edu.cmu.tetrad.search.CpcStable')(IndTest)
        cpcstable.setDepth(depth)
        cpcstable.setAggressivelyPreventCycles(aggressivelyPreventCycles)
        cpcstable.setVerbose(verbose)
        
        if priorKnowledge is not None:    
            cpcstable.setKnowledge(priorKnowledge)
            
        tetradGraph = cpcstable.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(tetradGraph)
        self.graph = pycausal.generatePyDotGraph(self.nodes,self.edges)

    def getDot(self):
        return self.graph
        
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):    
        return self.edges    
    
class bayesEst():
    
    graph = None
    nodes = []
    edges = []
    dag = None
    bayesPm = None
    bayesIm = None
    
    def __init__(self, df, depth = 3, significance = 0.05, verbose = False, priorKnowledge = None):
        tetradData = pycausal.loadDiscreteData(df)
        IndTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, significance)
        
        cpc = javabridge.JClassWrapper('edu.cmu.tetrad.search.Cpc')(IndTest)
        cpc.setDepth(depth)
        cpc.setVerbose(verbose)
        
        if priorKnowledge is not None:    
            cpc.setKnowledge(priorKnowledge)
            
        tetradGraph = cpc.search()
        dags = javabridge.JClassWrapper('edu.cmu.tetrad.search.DagInPatternIterator')(tetradGraph)
        dagGraph = dags.next()
        dag = javabridge.JClassWrapper('edu.cmu.tetrad.graph.Dag')(dagGraph)

        pm = javabridge.JClassWrapper('edu.cmu.tetrad.bayes.BayesPm')(dag)
        est = javabridge.JClassWrapper('edu.cmu.tetrad.bayes.MlBayesEstimator')()
        im = est.estimate(pm, tetradData)

        self.nodes = pycausal.extractTetradGraphNodes(dag)
        self.edges = pycausal.extractTetradGraphEdges(dag)
        self.graph = pycausal.generatePyDotGraph(self.nodes,self.edges)
        self.dag = dag
        self.bayesPm = pm
        self.bayesIm = im
        
    def getDot(self):
        return self.graph
        
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):    
        return self.edges    
    
    def getDag(self):
        return self.dag
    
    def getBayesPm(self):
        return self.bayesPm
    
    def getBayesIm(self):
        return self.bayesIm
    
class randomDag():
    
    graph = None
    nodes = []
    edges = []
    dag = None
    
    def __init__(self, df, seed = None, numNodes = 10, numEdges = 10):
        if seed is not None:
            RandomUtil = javabridge.static_call("edu/cmu/tetrad/util/RandomUtil","getInstance","()Ledu/cmu/tetrad/util/RandomUtil;")
            javabridge.call(RandomUtil, "setSeed", "(J)V", seed)
        
        dag = None
        initEdges = -1
        while initEdges < numEdges:
            graph = javabridge.static_call("edu/cmu/tetrad/graph/GraphUtils","randomGraph","(IIIIIIZ)Ledu/cmu/tetrad/graph/Graph;",numNodes,0,numEdges,30,15,15,False)
            dag = javabridge.JClassWrapper("edu.cmu.tetrad.graph.Dag")(graph)
            initEdges = dag.getNumEdges()
            
        self.nodes = pycausal.extractTetradGraphNodes(dag)
        self.edges = pycausal.extractTetradGraphEdges(dag)
        self.graph = pycausal.generatePyDotGraph(self.nodes,self.edges)
        self.dag = dag
        
    def getDot(self):
        return self.graph
        
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):    
        return self.edges    
    
    def getDag(self):
        return self.dag
    
