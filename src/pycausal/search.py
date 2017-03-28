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

class fgesDiscrete():
    
    graph = None
    nodes = []
    edges = []
    
    def __init__(self, df, structurePrior = 1.0, samplePrior = 1.0, maxDegree = 3, faithfulnessAssumed = True, numofthreads = 2, verbose = False, priorKnowledge = None):
        
        tetradData = pycausal.loadDiscreteData(df)
        
        score = javabridge.JClassWrapper('edu.cmu.tetrad.search.BDeuScore')(tetradData)
        score.setStructurePrior(structurePrior)
        score.setSamplePrior(samplePrior)
        
        fges = javabridge.JClassWrapper('edu.cmu.tetrad.search.Fgs')(score)
        fges.setMaxDegree(maxDegree)
        fges.setNumPatternsToStore(0)
        fges.setFaithfulnessAssumed(faithfulnessAssumed)
        fges.setParallelism(numofthreads)
        fges.setVerbose(verbose)
        
        if priorKnowledge is not None:    
            fges.setKnowledge(priorKnowledge)
            
        tetradGraph = fges.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(tetradGraph)            
        self.graph = pycausal.generatePyDotGraph(self.nodes,self.edges)
        
        
    def getDot(self):
        return self.graph
    
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):    
        return self.edges

    
class fges():
    
    graph = None
    nodes = []
    edges = []
    
    def __init__(self, df, penaltydiscount = 4, maxDegree = 3, faithfulnessAssumed = True, numofthreads = 2, verbose = False, priorKnowledge = None):
            
        tetradData = pycausal.loadContinuousData(df)

        score = javabridge.JClassWrapper('edu.cmu.tetrad.search.SemBicScore')(tetradData)
        score.setPenaltyDiscount(penaltydiscount) # set to 2 if variable# <= 50 otherwise set it to 4
        
        fges = javabridge.JClassWrapper('edu.cmu.tetrad.search.Fgs')(score)
        fges.setMaxDegree(maxDegree)
        fges.setNumPatternsToStore(0)
        fges.setFaithfulnessAssumed(faithfulnessAssumed)
        fges.setParallelism(numofthreads)
        fges.setVerbose(verbose)
        
        if priorKnowledge is not None:    
            fges.setKnowledge(priorKnowledge)
            
        tetradGraph = fges.search()
        
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
        indTest = None
        
        if(continuous):
            tetradData = pycausal.loadContinuousData(df)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, significance)
        else:
            tetradData = pycausal.loadDiscreteData(df)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, significance)
        
        fci = javabridge.JClassWrapper('edu.cmu.tetrad.search.Fci')(indTest)
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
        indTest = None
        
        if(continuous):
            tetradData = pycausal.loadContinuousData(df)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, significance)
        else:
            tetradData = pycausal.loadDiscreteData(df)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, significance)
        
        cfci = javabridge.JClassWrapper('edu.cmu.tetrad.search.Cfci')(indTest)
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
        
class rfci():
    
    nodes = []
    edges = []
    
    def __init__(self, df, continuous = True, depth = 3, significance = 0.05, completeRuleSetUsed = False, verbose = False, priorKnowledge = None):
        indTest = None
        
        if(continuous):
            tetradData = pycausal.loadContinuousData(df)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, significance)
        else:
            tetradData = pycausal.loadDiscreteData(df)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, significance)
        
        rfci = javabridge.JClassWrapper('edu.cmu.tetrad.search.Rfci')(indTest)
        rfci.setDepth(depth)
        rfci.setCompleteRuleSetUsed(completeRuleSetUsed)
        rfci.setVerbose(verbose)
        
        if priorKnowledge is not None:    
            rfci.setKnowledge(priorKnowledge)
            
        tetradGraph = rfci.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(tetradGraph) 
        
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):    
        return self.edges
        
class gfciDiscrete():
    
    nodes = []
    edges = []
    
    def __init__(self, df, structurePrior = 1.0, samplePrior = 1.0, maxDegree = 3, maxPathLength = -1, significance = 0.05, completeRuleSetUsed = False, faithfulnessAssumed = True, verbose = False, priorKnowledge = None):
        tetradData = pycausal.loadDiscreteData(df)
        
        indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, significance)
        
        score = javabridge.JClassWrapper('edu.cmu.tetrad.search.BDeuScore')(tetradData)
        score.setStructurePrior(structurePrior)
        score.setSamplePrior(samplePrior)
        
        gfci = javabridge.JClassWrapper('edu.cmu.tetrad.search.GFci')(indTest, score)
        gfci.setMaxDegree(maxDegree)
        gfci.setMaxPathLength(maxPathLength)
        gfci.setCompleteRuleSetUsed(completeRuleSetUsed)
        gfci.setFaithfulnessAssumed(faithfulnessAssumed)
        gfci.setVerbose(verbose)
        
        if priorKnowledge is not None:    
            gfci.setKnowledge(priorKnowledge)
            
        tetradGraph = gfci.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(tetradGraph) 
        
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):    
        return self.edges        
        
class gfci():
    
    nodes = []
    edges = []
    
    def __init__(self, df, penaltydiscount = 2, maxDegree = 3, maxPathLength = -1, significance = 0.05, completeRuleSetUsed = False, faithfulnessAssumed = True, verbose = False, priorKnowledge = None):
        tetradData = pycausal.loadContinuousData(df)
        
        indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, significance)
        
        score = javabridge.JClassWrapper('edu.cmu.tetrad.search.SemBicScore')(tetradData)
        score.setPenaltyDiscount(penaltydiscount) # set to 2 if variable# <= 50 otherwise set it to 4
        
        gfci = javabridge.JClassWrapper('edu.cmu.tetrad.search.GFci')(indTest, score)
        gfci.setMaxDegree(maxDegree)
        gfci.setMaxPathLength(maxPathLength)
        gfci.setCompleteRuleSetUsed(completeRuleSetUsed)
        gfci.setFaithfulnessAssumed(faithfulnessAssumed)
        gfci.setVerbose(verbose)
        
        if priorKnowledge is not None:    
            gfci.setKnowledge(priorKnowledge)
            
        tetradGraph = gfci.search()
        
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
        indTest = None
        
        if(continuous):
            tetradData = pycausal.loadContinuousData(df)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, significance)
        else:
            tetradData = pycausal.loadDiscreteData(df)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, significance)
        
        ccd = javabridge.JClassWrapper('edu.cmu.tetrad.search.Ccd')(indTest)
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
        indTest = None
    
        if(continuous):
            tetradData = pycausal.loadContinuousData(df)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, significance)
        else:
            tetradData = pycausal.loadDiscreteData(df)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, significance)
        
        pc = javabridge.JClassWrapper('edu.cmu.tetrad.search.Pc')(indTest)
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
    
class pcmax():
    
    graph = None
    nodes = []
    edges = []
    
    def __init__(self, df, continuous = True, depth = 3, maxPathLength = 3, useHeuristic = True, significance = 0.05, verbose = False, priorKnowledge = None):
        indTest = None
    
        if(continuous):
            tetradData = pycausal.loadContinuousData(df)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, significance)
        else:
            tetradData = pycausal.loadDiscreteData(df)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, significance)
        
        pcmax = javabridge.JClassWrapper('edu.cmu.tetrad.search.PcMax')(indTest)
        pcmax.setDepth(depth)
        pcmax.setMaxPathLength(maxPathLength)
        pcmax.setUseHeuristic(useHeuristic)
        pcmax.setVerbose(verbose)
        
        if priorKnowledge is not None:    
            pcmax.setKnowledge(priorKnowledge)
            
        tetradGraph = pcmax.search()
        
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
        indTest = None
    
        if(continuous):
            tetradData = pycausal.loadContinuousData(df)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, significance)
        else:
            tetradData = pycausal.loadDiscreteData(df)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, significance)
        
        pcstable = javabridge.JClassWrapper('edu.cmu.tetrad.search.PcStable')(indTest)
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
        indTest = None
    
        if(continuous):
            tetradData = pycausal.loadContinuousData(df)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, significance)
        else:
            tetradData = pycausal.loadDiscreteData(df)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, significance)
        
        cpc = javabridge.JClassWrapper('edu.cmu.tetrad.search.Cpc')(indTest)
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
        indTest = None
    
        if(continuous):
            tetradData = pycausal.loadContinuousData(df)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, significance)
        else:
            tetradData = pycausal.loadDiscreteData(df)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, significance)
        
        cpcstable = javabridge.JClassWrapper('edu.cmu.tetrad.search.CpcStable')(indTest)
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
        indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, significance)
        
        cpc = javabridge.JClassWrapper('edu.cmu.tetrad.search.Cpc')(indTest)
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
    
    def __init__(self, seed = None, numNodes = 10, numEdges = 10):
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
    
