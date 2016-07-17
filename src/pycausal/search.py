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
    
    def __init__(self, df, penaltydiscount = 4, depth = 3, heuristicSpeedup = True, numofthreads = 2, verbose = False, priorKnowledge = None):
            
        tetradData = pycausal.loadContinuousData(df)

        score = javabridge.JClassWrapper('edu.cmu.tetrad.search.SemBicScore')(tetradData)
        score.setPenaltyDiscount(penaltydiscount) # set to 2 if variable# <= 50 otherwise set it to 4
        
        fgs = javabridge.JClassWrapper('edu.cmu.tetrad.search.Fgs')(score)
        fgs.setDepth(depth)#-1
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
        fci.setDepth(depth)#-1
        fci.setPossibleDsepSearchDone(!noDSepSearch)
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
        