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

class fofc():

    graph = None
    nodes = []
    edges = []
    
    def __init__(self, df, testType = 'TETRAD_WISHART', fofcAlgorithm = 'GAP', alpha = .01):
        tetradData = pycausal.loadContinuousData(df)
        
        testType = javabridge.get_static_field('edu/cmu/tetrad/search/TestType',testType,'Ledu/cmu/tetrad/search/TestType;')
        fofcAlgorithm = javabridge.get_static_field('edu/cmu/tetrad/search/FindOneFactorClusters/Algorithm', fofcAlgorithm, 'Ledu/cmu/tetrad/search/FindOneFactorClusters/Algorithm;')
    
        fofc = javabridge.JClassWrapper('edu.cmu.tetrad.search.FindOneFactorClusters')(tetradData, testType, fofcAlgorithm, alpha)
    
        tetradGraph = fofc.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(tetradGraph)
        self.graph = pycausal.generatePyDotGraph(self.nodes,self.edges)
        
    def getDot(self):
        return self.graph
    
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):
        return self.edges
    
class dm():
    
    graph = None
    nodes = []
    edges = []
    
    def __init__(self, inputs, outputs, useGES = True, df, trueInputs, alphaPC = .05, alphaSober = .05, gesDiscount = 10, verbose = False, minDiscount = 4):

        orig_columns = df.columns.values
        new_columns = df.columns.values
        col_no = 0
        for col in df.columns:
            new_columns[col_no] = 'X' + str(col_no)
            col_no = col_no + 1
        df.columns = new_columns
        
        tetradData = pycausal.loadContinuousData(df)
        
        dm = javabridge.JClassWrapper('edu.cmu.tetrad.search.DMSearch')()
        dm.setInputs(inputs)
        dm.setOutputs(outputs)
        dm.setTrueInputs(trueInputs)
        dm.setData(tetradData)
        dm.setVerbose(verbose)
        dm.setAlphaSober(alphaSober)
        
        if useGES == True:
            dm.setAlphaPC(alphaPC)
        else:
            dm.setDiscount(gesDiscount)
            dm.setMinDiscount(minDiscount)
            
        tetradGraph = dm.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(tetradGraph, orig_columns, new_columns)
        self.edges = pycausal.extractTetradGraphEdges(tetradGraph, orig_columns, new_columns)
        self.graph = pycausal.generatePyDotGraph(self.nodes,self.edges)
        
    def getDot(self):
        return self.graph
    
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):
        return self.edges

class imagesBDeu():

    graph = None
    nodes = []
    edges = []

    def __init__(self, dfs, structurePrior = 1.0, samplePrior = 1.0, maxDegree = 3, faithfulnessAssumed = True, verbose = False, priorKnowledge = None):
        datasets = javabridge.JClassWrapper('java.util.ArrayList')()
        for idx in range(len(dfs)):
            df = dfs[idx]
            tetradData = pycausal.loadDiscreteData(df)
            datasets.add(tetradData)
        
        score = javabridge.JClassWrapper('edu.cmu.tetrad.search.BdeuScoreImages')(datasets)
        score.setStructurePrior(structurePrior)
        score.setSamplePrior(samplePrior)
        
        fges = javabridge.JClassWrapper('edu.cmu.tetrad.search.Fges')(score)
        fges.setMaxDegree(maxDegree)
        fges.setNumPatternsToStore(0)
        fges.setFaithfulnessAssumed(faithfulnessAssumed)
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

class imagesSemBic():

    graph = None
    nodes = []
    edges = []
    
    def __init__(self, dfs, penaltydiscount = 4, maxDegree = 3, faithfulnessAssumed = True, verbose = False, priorKnowledge = None):
        datasets = javabridge.JClassWrapper('java.util.ArrayList')()
        for idx in range(len(dfs)):
            df = dfs[idx]
            tetradData = pycausal.loadContinuousData(df)
            datasets.add(tetradData)
        
        score = javabridge.JClassWrapper('edu.cmu.tetrad.search.SemBicScoreImages')(datasets)
        score.setPenaltyDiscount(penaltydiscount) # set to 2 if variable# <= 50 otherwise set it to 4
        
        fges = javabridge.JClassWrapper('edu.cmu.tetrad.search.Fges')(score)
        fges.setMaxDegree(maxDegree)
        fges.setNumPatternsToStore(0)
        fges.setFaithfulnessAssumed(faithfulnessAssumed)
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

class bootstrapFgesMixed():
    
    graph = None
    nodes = []
    edges = []
    
    def __init__(self, df, numCategoriesToDiscretize = 4, penaltydiscount = 4, structurePrior = 1.0, maxDegree = 3, faithfulnessAssumed = True, numBootstrapSamples = 10, ensembleMethod = 'Highest', verbose = False, priorKnowledge = None):
        
        tetradData = pycausal.loadMixedData(df, numCategoriesToDiscretize)
        
        parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
        parameters.set('penaltyDiscount', penaltydiscount)
        parameters.set('structurePrior', structurePrior)
        parameters.set('faithfulnessAssumed', faithfulnessAssumed)
        parameters.set('maxDegree', maxDegree)
        parameters.set('numPatternsToStore', 0)
        parameters.set('verbose', verbose)
        
        algoName = javabridge.get_static_field('edu/pitt/dbmi/algo/bootstrap/BootstrapAlgName','FGES','Ledu/pitt/dbmi/algo/bootstrap/BootstrapAlgName;')
        fges = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.BootstrapTest')(tetradData, algoName)
        fges.setNumBootstrapSamples(numBootstrapSamples)
        fges.setVerbose(verbose)
        fges.setParameters(parameters)
        fges.setEdgeEnsemble(ensembleMethod)
        fges.setParallelMode(False)
        
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

class bootstrapFgesDiscrete():

    graph = None
    nodes = []
    edges = []
    
    def __init__(self, df, structurePrior = 1.0, samplePrior = 1.0, maxDegree = 3, faithfulnessAssumed = True, numBootstrapSamples = 10, ensembleMethod = 'Highest', verbose = False, priorKnowledge = None):
        
        tetradData = pycausal.loadDiscreteData(df)
        
        parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
        parameters.set('structurePrior', structurePrior)
        parameters.set('samplePrior', samplePrior)
        parameters.set('faithfulnessAssumed', faithfulnessAssumed)
        parameters.set('maxDegree', maxDegree)
        parameters.set('numPatternsToStore', 0)
        parameters.set('verbose', verbose)
        
        algoName = javabridge.get_static_field('edu/pitt/dbmi/algo/bootstrap/BootstrapAlgName','FGES','Ledu/pitt/dbmi/algo/bootstrap/BootstrapAlgName;')
        fges = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.BootstrapTest')(tetradData, algoName)
        fges.setNumBootstrapSamples(numBootstrapSamples)
        fges.setVerbose(verbose)
        fges.setParameters(parameters)
        fges.setEdgeEnsemble(ensembleMethod)
        fges.setParallelMode(False)
        
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

class bootstrapFges():
    
    graph = None
    nodes = []
    edges = []
    
    def __init__(self, df, penaltydiscount = 4, maxDegree = 3, faithfulnessAssumed = True, numBootstrapSamples = 10, ensembleMethod = 'Highest', verbose = False, priorKnowledge = None):
        
        tetradData = pycausal.loadContinuousData(df, outputDataset = True)
        
        parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
        parameters.set('penaltyDiscount', penaltydiscount)
        parameters.set('maxDegree', maxDegree)
        parameters.set('faithfulnessAssumed', faithfulnessAssumed)
        parameters.set('numPatternsToStore', 0)
        parameters.set('verbose', verbose)
        
        algoName = javabridge.get_static_field('edu/pitt/dbmi/algo/bootstrap/BootstrapAlgName','FGES','Ledu/pitt/dbmi/algo/bootstrap/BootstrapAlgName;')
        fges = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.BootstrapTest')(tetradData, algoName)
        fges.setNumBootstrapSamples(numBootstrapSamples)
        fges.setVerbose(verbose)
        fges.setParameters(parameters)
        fges.setEdgeEnsemble(ensembleMethod)
        fges.setParallelMode(False)
        
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

class bootstrapGfciMixed():
    
    nodes = []
    edges = []
    
    def __init__(self, df, numCategoriesToDiscretize = 4, penaltydiscount = 4, structurePrior = 1.0, maxDegree = 3, maxPathLength = -1, significance = 0.05, completeRuleSetUsed = False, faithfulnessAssumed = True, numBootstrapSamples = 10, ensembleMethod = 'Highest', verbose = False, priorKnowledge = None):
        
        tetradData = pycausal.loadMixedData(df, numCategoriesToDiscretize)
        
        parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
        parameters.set('penaltyDiscount', penaltydiscount)
        parameters.set('structurePrior', structurePrior)
        parameters.set('faithfulnessAssumed', faithfulnessAssumed)
        parameters.set('maxDegree', maxDegree)
        parameters.set('maxPathLength', maxPathLength)
        parameters.set('significance', significance)
        parameters.set('completeRuleSetUsed', completeRuleSetUsed)
        parameters.set('verbose', verbose)
        
        algoName = javabridge.get_static_field('edu/pitt/dbmi/algo/bootstrap/BootstrapAlgName','GFCI','Ledu/pitt/dbmi/algo/bootstrap/BootstrapAlgName;')
        gfci = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.BootstrapTest')(tetradData, algoName)
        gfci.setNumBootstrapSamples(numBootstrapSamples)
        gfci.setVerbose(verbose)
        gfci.setParameters(parameters)
        gfci.setEdgeEnsemble(ensembleMethod)
        gfci.setParallelMode(False)
        
        if priorKnowledge is not None:
            gfci.setKnowledge(priorKnowledge)
        
        tetradGraph = gfci.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(tetradGraph)
    
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):
        return self.edges

class bootstrapGfciDiscrete():
    
    nodes = []
    edges = []
    
    def __init__(self, df, structurePrior = 1.0, samplePrior = 1.0, maxDegree = 3, maxPathLength = -1, significance = 0.05, completeRuleSetUsed = False, faithfulnessAssumed = True, numBootstrapSamples = 10, ensembleMethod = 'Highest', verbose = False, priorKnowledge = None):

        tetradData = pycausal.loadDiscreteData(df)
        
        parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
        parameters.set('structurePrior', structurePrior)
        parameters.set('samplePrior', samplePrior)
        parameters.set('faithfulnessAssumed', faithfulnessAssumed)
        parameters.set('maxDegree', maxDegree)
        parameters.set('maxPathLength', maxPathLength)
        parameters.set('significance', significance)
        parameters.set('completeRuleSetUsed', completeRuleSetUsed)
        parameters.set('verbose', verbose)
        
        algoName = javabridge.get_static_field('edu/pitt/dbmi/algo/bootstrap/BootstrapAlgName','GFCI','Ledu/pitt/dbmi/algo/bootstrap/BootstrapAlgName;')
        gfci = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.BootstrapTest')(tetradData, algoName)
        gfci.setNumBootstrapSamples(numBootstrapSamples)
        gfci.setVerbose(verbose)
        gfci.setParameters(parameters)
        gfci.setEdgeEnsemble(ensembleMethod)
        gfci.setParallelMode(False)
        
        if priorKnowledge is not None:
            gfci.setKnowledge(priorKnowledge)
        
        tetradGraph = gfci.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(tetradGraph)

    def getNodes(self):
        return self.nodes
    
    def getEdges(self):
        return self.edges

class bootstrapGfci():
    
    nodes = []
    edges = []
    
    def __init__(self, df, penaltydiscount = 4, maxDegree = 3, maxPathLength = -1, significance = 0.05, completeRuleSetUsed = False, faithfulnessAssumed = True, numBootstrapSamples = 10, ensembleMethod = 'Highest', verbose = False, priorKnowledge = None):

        tetradData = pycausal.loadContinuousData(df, outputDataset = True)
        
        parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
        parameters.set('penaltyDiscount', penaltydiscount)
        parameters.set('maxDegree', maxDegree)
        parameters.set('faithfulnessAssumed', faithfulnessAssumed)
        parameters.set('maxPathLength', maxPathLength)
        parameters.set('maxDegree', maxDegree)
        parameters.set('significance', significance)
        parameters.set('completeRuleSetUsed', completeRuleSetUsed)
        parameters.set('verbose', verbose)
        
        algoName = javabridge.get_static_field('edu/pitt/dbmi/algo/bootstrap/BootstrapAlgName','GFCI','Ledu/pitt/dbmi/algo/bootstrap/BootstrapAlgName;')
        gfci = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.BootstrapTest')(tetradData, algoName)
        gfci.setNumBootstrapSamples(numBootstrapSamples)
        gfci.setVerbose(verbose)
        gfci.setParameters(parameters)
        gfci.setEdgeEnsemble(ensembleMethod)
        gfci.setParallelMode(False)
        
        if priorKnowledge is not None:
            gfci.setKnowledge(priorKnowledge)
        
        tetradGraph = gfci.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(tetradGraph)

    def getNodes(self):
        return self.nodes
    
    def getEdges(self):
        return self.edges

class bootstrapRfciMixed():
    
    nodes = []
    edges = []
    
    def __init__(self, df, numCategoriesToDiscretize = 4, depth = 3, maxPathLength = -1, significance = 0.05, completeRuleSetUsed = False, numBootstrapSamples = 10, ensembleMethod = 'Highest', verbose = False, priorKnowledge = None):
        
        tetradData = pycausal.loadMixedData(df, numCategoriesToDiscretize)
        
        parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
        parameters.set('depth', depth)
        parameters.set('maxPathLength', maxPathLength)
        parameters.set('significance', significance)
        parameters.set('completeRuleSetUsed', completeRuleSetUsed)
        parameters.set('verbose', verbose)
        
        algoName = javabridge.get_static_field('edu/pitt/dbmi/algo/bootstrap/BootstrapAlgName','RFCI','Ledu/pitt/dbmi/algo/bootstrap/BootstrapAlgName;')
        rfci = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.BootstrapTest')(tetradData, algoName)
        rfci.setNumBootstrapSamples(numBootstrapSamples)
        rfci.setVerbose(verbose)
        rfci.setParameters(parameters)
        rfci.setEdgeEnsemble(ensembleMethod)
        rfci.setParallelMode(False)
        
        if priorKnowledge is not None:
            rfci.setKnowledge(priorKnowledge)
        
        tetradGraph = rfci.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(tetradGraph)
    
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):
        return self.edges

class bootstrapRfciDiscrete():
    
    nodes = []
    edges = []
    
    def __init__(self, df, depth = 3, maxPathLength = -1, significance = 0.05, completeRuleSetUsed = False, numBootstrapSamples = 10, ensembleMethod = 'Highest', verbose = False, priorKnowledge = None):
        
        tetradData = pycausal.loadDiscreteData(df)
    
        parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
        parameters.set('depth', depth)
        parameters.set('maxPathLength', maxPathLength)
        parameters.set('significance', significance)
        parameters.set('completeRuleSetUsed', completeRuleSetUsed)
        parameters.set('verbose', verbose)
        
        algoName = javabridge.get_static_field('edu/pitt/dbmi/algo/bootstrap/BootstrapAlgName','RFCI','Ledu/pitt/dbmi/algo/bootstrap/BootstrapAlgName;')
        rfci = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.BootstrapTest')(tetradData, algoName)
        rfci.setNumBootstrapSamples(numBootstrapSamples)
        rfci.setVerbose(verbose)
        rfci.setParameters(parameters)
        rfci.setEdgeEnsemble(ensembleMethod)
        rfci.setParallelMode(False)
        
        if priorKnowledge is not None:
            rfci.setKnowledge(priorKnowledge)

        tetradGraph = rfci.search()
    
        self.nodes = pycausal.extractTetradGraphNodes(tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(tetradGraph)
    
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):
        return self.edges

class bootstrapRfci():
    
    nodes = []
    edges = []
    
    def __init__(self, df, depth = 3, maxPathLength = -1, significance = 0.05, completeRuleSetUsed = False, numBootstrapSamples = 10, ensembleMethod = 'Highest', verbose = False, priorKnowledge = None):

        tetradData = pycausal.loadContinuousData(df, outputDataset = True)
        
        parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
        parameters.set('depth', depth)
        parameters.set('maxPathLength', maxPathLength)
        parameters.set('significance', significance)
        parameters.set('completeRuleSetUsed', completeRuleSetUsed)
        parameters.set('verbose', verbose)

        algoName = javabridge.get_static_field('edu/pitt/dbmi/algo/bootstrap/BootstrapAlgName','RFCI','Ledu/pitt/dbmi/algo/bootstrap/BootstrapAlgName;')
        rfci = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.BootstrapTest')(tetradData, algoName)
        rfci.setNumBootstrapSamples(numBootstrapSamples)
        rfci.setVerbose(verbose)
        rfci.setParameters(parameters)
        rfci.setEdgeEnsemble(ensembleMethod)
        rfci.setParallelMode(False)
        
        if priorKnowledge is not None:
            rfci.setKnowledge(priorKnowledge)
        
        tetradGraph = rfci.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(tetradGraph)

    def getNodes(self):
        return self.nodes
    
    def getEdges(self):
        return self.edges

class fasMixed():
    
    nodes = []
    edges = []
    
    def __init__(self, df, numCategoriesToDiscretize = 4, depth = 3, significance = 0.05, sepsetsReturnEmptyIfNotFixed = False, verbose = False, priorKnowledge = None):
        
        tetradData = pycausal.loadMixedData(df, numCategoriesToDiscretize)
        indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestConditionalGaussianLRT')(tetradData, significance)
        
        fas = javabridge.JClassWrapper('edu.cmu.tetrad.search.Fas')(indTest)
        fas.setDepth(depth)
        fas.setSepsetsReturnEmptyIfNotFixed(sepsetsReturnEmptyIfNotFixed)
        fas.setVerbose(verbose)
        
        if priorKnowledge is not None:
            fas.setKnowledge(priorKnowledge)
        
        tetradGraph = fas.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(tetradGraph)
    
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):
        return self.edges

class fasDiscrete():
    
    nodes = []
    edges = []
    
    def __init__(self, df, depth = 3, significance = 0.05, sepsetsReturnEmptyIfNotFixed = False, verbose = False, priorKnowledge = None):
        
        tetradData = pycausal.loadDiscreteData(df)
        indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, significance)
        
        fas = javabridge.JClassWrapper('edu.cmu.tetrad.search.Fas')(indTest)
        fas.setDepth(depth)
        fas.setSepsetsReturnEmptyIfNotFixed(sepsetsReturnEmptyIfNotFixed)
        fas.setVerbose(verbose)
        
        if priorKnowledge is not None:
            fas.setKnowledge(priorKnowledge)
        
        tetradGraph = fas.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(tetradGraph)
    
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):
        return self.edges

class fas():
    
    nodes = []
    edges = []
    
    def __init__(self, df, depth = 3, significance = 0.05, sepsetsReturnEmptyIfNotFixed = False, verbose = False, priorKnowledge = None):
        
        tetradData = pycausal.loadContinuousData(df)
        indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, significance)
        
        fas = javabridge.JClassWrapper('edu.cmu.tetrad.search.Fas')(indTest)
        fas.setDepth(depth)
        fas.setSepsetsReturnEmptyIfNotFixed(sepsetsReturnEmptyIfNotFixed)
        fas.setVerbose(verbose)
        
        if priorKnowledge is not None:
            fas.setKnowledge(priorKnowledge)
        
        tetradGraph = fas.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(tetradGraph)

    def getNodes(self):
        return self.nodes
    
    def getEdges(self):
        return self.edges

class fgesMixed():
    
    graph = None
    nodes = []
    edges = []
    
    def __init__(self, df, numCategoriesToDiscretize = 4, penaltydiscount = 4, structurePrior = 1.0, maxDegree = 3, faithfulnessAssumed = True, verbose = False, priorKnowledge = None):
        
        tetradData = pycausal.loadMixedData(df, numCategoriesToDiscretize)
        
        score = javabridge.JClassWrapper('edu.cmu.tetrad.search.ConditionalGaussianScore')(tetradData, structurePrior, True)
        score.setPenaltyDiscount(penaltydiscount) # set to 2 if variable# <= 50 otherwise set it to 4
        
        fges = javabridge.JClassWrapper('edu.cmu.tetrad.search.Fges')(score)
        fges.setMaxDegree(maxDegree)
        fges.setNumPatternsToStore(0)
        fges.setFaithfulnessAssumed(faithfulnessAssumed)
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

class fgesDiscrete():
    
    graph = None
    nodes = []
    edges = []
    
    def __init__(self, df, structurePrior = 1.0, samplePrior = 1.0, maxDegree = 3, faithfulnessAssumed = True, verbose = False, priorKnowledge = None):
        
        tetradData = pycausal.loadDiscreteData(df)
        
        score = javabridge.JClassWrapper('edu.cmu.tetrad.search.BDeuScore')(tetradData)
        score.setStructurePrior(structurePrior)
        score.setSamplePrior(samplePrior)
        
        fges = javabridge.JClassWrapper('edu.cmu.tetrad.search.Fges')(score)
        fges.setMaxDegree(maxDegree)
        fges.setNumPatternsToStore(0)
        fges.setFaithfulnessAssumed(faithfulnessAssumed)
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
    
    def __init__(self, df, penaltydiscount = 4, maxDegree = 3, faithfulnessAssumed = True, verbose = False, priorKnowledge = None):
            
        tetradData = pycausal.loadContinuousData(df)

        score = javabridge.JClassWrapper('edu.cmu.tetrad.search.SemBicScore')(tetradData)
        score.setPenaltyDiscount(penaltydiscount) # set to 2 if variable# <= 50 otherwise set it to 4
        
        fges = javabridge.JClassWrapper('edu.cmu.tetrad.search.Fges')(score)
        fges.setMaxDegree(maxDegree)
        fges.setNumPatternsToStore(0)
        fges.setFaithfulnessAssumed(faithfulnessAssumed)
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
        
class rfciMixed():
    
    nodes = []
    edges = []
    
    def __init__(self, df, numCategoriesToDiscretize = 4, depth = 3, maxPathLength = -1, significance = 0.05, completeRuleSetUsed = False, verbose = False, priorKnowledge = None):
        tetradData = pycausal.loadMixedData(df, numCategoriesToDiscretize)
        
        indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestConditionalGaussianLRT')(tetradData, significance)
        
        rfci = javabridge.JClassWrapper('edu.cmu.tetrad.search.Rfci')(indTest)
        rfci.setDepth(depth)
        rfci.setMaxPathLength(maxPathLength)
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

class rfciDiscrete():
    
    nodes = []
    edges = []
    
    def __init__(self, df, depth = 3, maxPathLength = -1, significance = 0.05, completeRuleSetUsed = False, verbose = False, priorKnowledge = None):
        tetradData = pycausal.loadDiscreteData(df)
        indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, significance)
        
        rfci = javabridge.JClassWrapper('edu.cmu.tetrad.search.Rfci')(indTest)
        rfci.setDepth(depth)
        rfci.setMaxPathLength(maxPathLength)
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

class rfci():
    
    nodes = []
    edges = []
    
    def __init__(self, df, depth = 3, maxPathLength = -1, significance = 0.05, completeRuleSetUsed = False, verbose = False, priorKnowledge = None):
        tetradData = pycausal.loadContinuousData(df)
        indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, significance)
        
        rfci = javabridge.JClassWrapper('edu.cmu.tetrad.search.Rfci')(indTest)
        rfci.setDepth(depth)
        rfci.setMaxPathLength(maxPathLength)
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
        
class gfciMixed():
    
    nodes = []
    edges = []
    
    def __init__(self, df, numCategoriesToDiscretize = 4, penaltydiscount = 4, structurePrior = 1.0, maxDegree = 3, maxPathLength = -1, significance = 0.05, completeRuleSetUsed = False, faithfulnessAssumed = True, verbose = False, priorKnowledge = None):
        tetradData = pycausal.loadMixedData(df, numCategoriesToDiscretize)
        
        indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestConditionalGaussianLRT')(tetradData, significance)
        
        score = javabridge.JClassWrapper('edu.cmu.tetrad.search.ConditionalGaussianScore')(tetradData, structurePrior, True)
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
    
    def __init__(self, df, penaltydiscount = 4, maxDegree = 3, maxPathLength = -1, significance = 0.05, completeRuleSetUsed = False, faithfulnessAssumed = True, verbose = False, priorKnowledge = None):
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
    
    def __init__(self, df, continuous = True, depth = 3, significance = 0.05, priorKnowledge = None):
        indTest = None
        
        if(continuous):
            tetradData = pycausal.loadContinuousData(df)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, significance)
        else:
            tetradData = pycausal.loadDiscreteData(df)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, significance)
        
        ccd = javabridge.JClassWrapper('edu.cmu.tetrad.search.Ccd')(indTest)
        ccd.setDepth(depth)
        
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
    
class pcstablemax():
    
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
        
        pcmax = javabridge.JClassWrapper('edu.cmu.tetrad.search.PcStableMax')(indTest)
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
    
