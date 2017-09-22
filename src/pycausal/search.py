'''
Created on Feb 17, 2016

@author: Chirayu Wongchokprasitti, PhD 
@email: chw20@pitt.edu
'''

import javabridge
import os
import glob
import numpy as np
import pydot
import random
import string
import tempfile

import pycausal

class fofc():

    graph = None
    nodes = []
    edges = []
    
    def __init__(self, df, testType = 'TETRAD_WISHART', fofcAlgorithm = 'GAP', alpha = 0.01, numBootstrap = -1, ensembleMethod = 'Highest'):
        
        fofc = None

        tetradData = pycausal.loadContinuousData(df)
        
        if numBootstrap < 1:
            testType = javabridge.get_static_field('edu/cmu/tetrad/search/TestType',
                                                   testType,
                                                   'Ledu/cmu/tetrad/search/TestType;')
            fofcAlgorithm = javabridge.get_static_field('edu/cmu/tetrad/search/FindOneFactorClusters$Algorithm', 
                                                fofcAlgorithm, 
                                                'Ledu/cmu/tetrad/search/FindOneFactorClusters$Algorithm;')

            fofc = javabridge.JClassWrapper('edu.cmu.tetrad.search.FindOneFactorClusters')(tetradData, testType, fofcAlgorithm, alpha)
        else:    
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.cluster.Fofc')()
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            useWishart = True
            if testType != 'TETRAD_WISHART':
                useWishart = False
            parameters.set('useWishart', useWishart)
            useGap = True
            if fofcAlgorithm != 'GAP':
                useGap = False
            parameters.set('useGap', useGap)
            parameters.set('alpha', alpha)
            
            fofc = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble;')
            fofc.setEdgeEnsemble(edgeEnsemble)
            fofc.setParameters(parameters)
            
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
    
    def __init__(self, df, inputs, outputs, trueInputs, useGES = True, alphaPC = 0.05, alphaSober = 0.05, gesDiscount = 10, verbose = False, minDiscount = 4):

        inputs = javabridge.get_env().make_int_array(np.array(inputs, np.int32))
        outputs = javabridge.get_env().make_int_array(np.array(outputs, np.int32))
        trueInputs = javabridge.get_env().make_int_array(np.array(trueInputs, np.int32))
        
        orig_columns = df.columns.values
        orig_columns = orig_columns.tolist()
        new_columns = df.columns.values
        col_no = 0
        for col in df.columns:
            new_columns[col_no] = 'X' + str(col_no)
            col_no = col_no + 1
        df.columns = new_columns
        new_columns = new_columns.tolist()
        
        tetradData = pycausal.loadContinuousData(df, outputDataset = True)
        
        dm = javabridge.JClassWrapper('edu.cmu.tetrad.search.DMSearch')()
        dm.setInputs(inputs)
        dm.setOutputs(outputs)
        dm.setTrueInputs(trueInputs)
        dm.setData(tetradData)
        dm.setVerbose(verbose)
        
        if useGES:
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

    def __init__(self, dfs, structurePrior = 1.0, samplePrior = 1.0, maxDegree = 3, faithfulnessAssumed = True, verbose = False, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
        datasets = javabridge.JClassWrapper('java.util.ArrayList')()
        
        for idx in range(len(dfs)):
            df = dfs[idx]
            tetradData = pycausal.loadDiscreteData(df)
            datasets.add(tetradData)
        
        fges = None
        
        if numBootstrap < 1:
            score = javabridge.JClassWrapper('edu.cmu.tetrad.search.BdeuScoreImages')(datasets)
            score.setStructurePrior(structurePrior)
            score.setSamplePrior(samplePrior)

            fges = javabridge.JClassWrapper('edu.cmu.tetrad.search.Fges')(score)
            fges.setMaxDegree(maxDegree)
            fges.setNumPatternsToStore(0)
            fges.setFaithfulnessAssumed(faithfulnessAssumed)
        else:
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.multi.ImagesBDeu')()
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('structurePrior', structurePrior)
            parameters.set('samplePrior', samplePrior)
            parameters.set('maxDegree', maxDegree)
            parameters.set('faithfulnessAssumed', faithfulnessAssumed)
            parameters.set('verbose', verbose)
            
            fges = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(datasets, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble;')
            fges.setEdgeEnsemble(edgeEnsemble)
            fges.setParameters(parameters)

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
    
    def __init__(self, dfs, penaltyDiscount = 4, maxDegree = 3, faithfulnessAssumed = True, verbose = False, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
        datasets = javabridge.JClassWrapper('java.util.ArrayList')()
        for idx in range(len(dfs)):
            df = dfs[idx]
            tetradData = pycausal.loadContinuousData(df)
            datasets.add(tetradData)
        
        fges = None
        
        if numBootstrap < 1:
            score = javabridge.JClassWrapper('edu.cmu.tetrad.search.SemBicScoreImages')(datasets)
            score.setPenaltyDiscount(penaltyDiscount) # set to 2 if variable# <= 50 otherwise set it to 4

            fges = javabridge.JClassWrapper('edu.cmu.tetrad.search.Fges')(score)
            fges.setMaxDegree(maxDegree)
            fges.setNumPatternsToStore(0)
            fges.setFaithfulnessAssumed(faithfulnessAssumed)
        else:
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.multi.ImagesSemBic')()
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('penaltyDiscount', penaltyDiscount)
            parameters.set('maxDegree', maxDegree)
            parameters.set('faithfulnessAssumed', faithfulnessAssumed)
            parameters.set('verbose', verbose)
            
            fges = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(datasets, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble;')
            fges.setEdgeEnsemble(edgeEnsemble)
            fges.setParameters(parameters)
        
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

class fasMixed():
    
    nodes = []
    edges = []
    
    def __init__(self, df, numCategoriesToDiscretize = 4, depth = 3, alpha = 0.05, verbose = False, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
        
        tetradData = pycausal.loadMixedData(df, numCategoriesToDiscretize)       
        indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestConditionalGaussianLRT')(tetradData, alpha)

        fas = None
        
        if numBootstrap < 1:
            fas = javabridge.JClassWrapper('edu.cmu.tetrad.search.Fas')(indTest)
            fas.setDepth(depth)
        else:
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.oracle.pattern.FAS')(indTest)
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('depth', depth)
            parameters.set('verbose', verbose)
            
            fas = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble;')
            fas.setEdgeEnsemble(edgeEnsemble)
            fas.setParameters(parameters)
            
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
    
    def __init__(self, df, depth = 3, alpha = 0.05, verbose = False, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
        
        tetradData = pycausal.loadDiscreteData(df)
        indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, alpha)
        
        fas = None
        
        if numBootstrap < 1:
            fas = javabridge.JClassWrapper('edu.cmu.tetrad.search.Fas')(indTest)
            fas.setDepth(depth)
        else:
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.oracle.pattern.FAS')(indTest)
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('depth', depth)
            parameters.set('verbose', verbose)
            
            fas = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble;')
            fas.setEdgeEnsemble(edgeEnsemble)
            fas.setParameters(parameters)
            
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
    
    def __init__(self, df, depth = 3, alpha = 0.05, verbose = False, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
        
        tetradData = pycausal.loadContinuousData(df)
        indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, alpha)
        
        fas = None
        
        if numBootstrap < 1:
            fas = javabridge.JClassWrapper('edu.cmu.tetrad.search.Fas')(indTest)
            fas.setDepth(depth)
        else:
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.oracle.pattern.FAS')(indTest)
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('depth', depth)
            parameters.set('verbose', verbose)
            
            fas = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble;')
            fas.setEdgeEnsemble(edgeEnsemble)
            fas.setParameters(parameters)
            
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
    
    def __init__(self, df, numCategoriesToDiscretize = 4, penaltyDiscount = 4, structurePrior = 1.0, maxDegree = 3, faithfulnessAssumed = True, verbose = False, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
        
        tetradData = pycausal.loadMixedData(df, numCategoriesToDiscretize)

        score = javabridge.JClassWrapper('edu.cmu.tetrad.search.ConditionalGaussianScore')(tetradData, structurePrior, True)
        score.setPenaltyDiscount(penaltyDiscount) # set to 2 if variable# <= 50 otherwise set it to 4

        fges = None
        
        if numBootstrap < 1:
            fges = javabridge.JClassWrapper('edu.cmu.tetrad.search.Fges')(score)
            fges.setMaxDegree(maxDegree)
            fges.setNumPatternsToStore(0)
            fges.setFaithfulnessAssumed(faithfulnessAssumed)
        else:
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.oracle.pattern.Fges')(score)
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('maxDegree', maxDegree)
            parameters.set('faithfulnessAssumed', faithfulnessAssumed)
            parameters.set('verbose', verbose)
            
            fges = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble;')
            fges.setEdgeEnsemble(edgeEnsemble)
            fges.setParameters(parameters)
            
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
    
    def __init__(self, df, structurePrior = 1.0, samplePrior = 1.0, maxDegree = 3, faithfulnessAssumed = True, verbose = False, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
        
        tetradData = pycausal.loadDiscreteData(df)
        
        score = javabridge.JClassWrapper('edu.cmu.tetrad.search.BDeuScore')(tetradData)
        score.setStructurePrior(structurePrior)
        score.setSamplePrior(samplePrior)
        
        fges = None
        
        if numBootstrap < 1:
            fges = javabridge.JClassWrapper('edu.cmu.tetrad.search.Fges')(score)
            fges.setMaxDegree(maxDegree)
            fges.setNumPatternsToStore(0)
            fges.setFaithfulnessAssumed(faithfulnessAssumed)
        else:
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.oracle.pattern.Fges')(score)
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('maxDegree', maxDegree)
            parameters.set('faithfulnessAssumed', faithfulnessAssumed)
            parameters.set('verbose', verbose)
            
            fges = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble;')
            fges.setEdgeEnsemble(edgeEnsemble)
            fges.setParameters(parameters)
            
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
    
    def __init__(self, df, penaltyDiscount = 4, maxDegree = 3, faithfulnessAssumed = True, verbose = False, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
            
        tetradData = pycausal.loadContinuousData(df)

        score = javabridge.JClassWrapper('edu.cmu.tetrad.search.SemBicScore')(tetradData)
        score.setPenaltyDiscount(penaltyDiscount) # set to 2 if variable# <= 50 otherwise set it to 4
        
        fges = None
        
        if numBootstrap < 1:
            fges = javabridge.JClassWrapper('edu.cmu.tetrad.search.Fges')(score)
            fges.setMaxDegree(maxDegree)
            fges.setNumPatternsToStore(0)
            fges.setFaithfulnessAssumed(faithfulnessAssumed)
        else:
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.oracle.pattern.Fges')(score)
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('maxDegree', maxDegree)
            parameters.set('faithfulnessAssumed', faithfulnessAssumed)
            parameters.set('verbose', verbose)
            
            fges = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble;')
            fges.setEdgeEnsemble(edgeEnsemble)
            fges.setParameters(parameters)
            
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
    
    def __init__(self, df, dataType = 0, numCategoriesToDiscretize = 4, depth = 3, alpha = 0.05, completeRuleSetUsed = False, verbose = False, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
        tetradData = None
        indTest = None
        
        # Continuous
        if dataType == 0:
            tetradData = pycausal.loadContinuousData(df)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, alpha)
        # Discrete
        elif dataType == 1:
            tetradData = pycausal.loadDiscreteData(df)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, alpha)
        # Mixed
        else:
            tetradData = pycausal.loadMixedData(df, numCategoriesToDiscretize)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestConditionalGaussianLRT')(tetradData, alpha)
            
        fci = None
        
        if numBootstrap < 1:
            fci = javabridge.JClassWrapper('edu.cmu.tetrad.search.Fci')(indTest)
            fci.setDepth(depth)
            fci.setCompleteRuleSetUsed(completeRuleSetUsed)
        else:
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.oracle.pag.Fci')(indTest)
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('depth', depth)
            parameters.set('completeRuleSetUsed', completeRuleSetUsed)
            parameters.set('verbose', verbose)
            
            fci = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble;')
            fci.setEdgeEnsemble(edgeEnsemble)
            fci.setParameters(parameters)
            
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
    
    def __init__(self, df, dataType = 0, numCategoriesToDiscretize = 4, depth = 3, alpha = 0.05, verbose = False, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
        tetradData = None
        indTest = None
        
        # Continuous
        if dataType == 0:
            tetradData = pycausal.loadContinuousData(df)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, alpha)
        # Discrete
        elif dataType == 1:
            tetradData = pycausal.loadDiscreteData(df)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, alpha)
        # Mixed
        else:
            tetradData = pycausal.loadMixedData(df, numCategoriesToDiscretize)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestConditionalGaussianLRT')(tetradData, alpha)
        
        cfci = None
        
        if numBootstrap < 1:
            cfci = javabridge.JClassWrapper('edu.cmu.tetrad.search.Cfci')(indTest)
            cfci.setDepth(depth)
        else:
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.oracle.pag.Cfci')(indTest)
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('depth', depth)
            parameters.set('verbose', verbose)
            
            cfci = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble;')
            cfci.setEdgeEnsemble(edgeEnsemble)
            cfci.setParameters(parameters)
            
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
    
    def __init__(self, df, numCategoriesToDiscretize = 4, depth = 3, maxPathLength = -1, alpha = 0.05, completeRuleSetUsed = False, verbose = False, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
        tetradData = pycausal.loadMixedData(df, numCategoriesToDiscretize)       
        indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestConditionalGaussianLRT')(tetradData, alpha)
        
        rfci = None
        
        if numBootstrap < 1:
            rfci = javabridge.JClassWrapper('edu.cmu.tetrad.search.Rfci')(indTest)
            rfci.setDepth(depth)
            rfci.setMaxPathLength(maxPathLength)
            rfci.setCompleteRuleSetUsed(completeRuleSetUsed)
        else:
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.oracle.pag.Rfci')(indTest)
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('depth', depth)
            parameters.set('maxPathLength', maxPathLength)
            parameters.set('completeRuleSetUsed', completeRuleSetUsed)
            parameters.set('verbose', verbose)
            
            rfci = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble;')
            rfci.setEdgeEnsemble(edgeEnsemble)
            rfci.setParameters(parameters)
            
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
    
    def __init__(self, df, depth = 3, maxPathLength = -1, alpha = 0.05, completeRuleSetUsed = False, verbose = False, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
        tetradData = pycausal.loadDiscreteData(df)
        indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, alpha)
        
        rfci = None
        
        if numBootstrap < 1:
            rfci = javabridge.JClassWrapper('edu.cmu.tetrad.search.Rfci')(indTest)
            rfci.setDepth(depth)
            rfci.setMaxPathLength(maxPathLength)
            rfci.setCompleteRuleSetUsed(completeRuleSetUsed)
        else:
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.oracle.pag.Rfci')(indTest)
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('depth', depth)
            parameters.set('maxPathLength', maxPathLength)
            parameters.set('completeRuleSetUsed', completeRuleSetUsed)
            parameters.set('verbose', verbose)
            
            rfci = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble;')
            rfci.setEdgeEnsemble(edgeEnsemble)
            rfci.setParameters(parameters)
        
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
    
    def __init__(self, df, depth = 3, maxPathLength = -1, alpha = 0.05, completeRuleSetUsed = False, verbose = False, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
        tetradData = pycausal.loadContinuousData(df)
        indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, alpha)
        
        rfci = None
        
        if numBootstrap < 1:
            rfci = javabridge.JClassWrapper('edu.cmu.tetrad.search.Rfci')(indTest)
            rfci.setDepth(depth)
            rfci.setMaxPathLength(maxPathLength)
            rfci.setCompleteRuleSetUsed(completeRuleSetUsed)
        else:
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.oracle.pag.Rfci')(indTest)
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('depth', depth)
            parameters.set('maxPathLength', maxPathLength)
            parameters.set('completeRuleSetUsed', completeRuleSetUsed)
            parameters.set('verbose', verbose)
            
            rfci = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble;')
            rfci.setEdgeEnsemble(edgeEnsemble)
            rfci.setParameters(parameters)
            
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
    
    def __init__(self, df, numCategoriesToDiscretize = 4, penaltyDiscount = 4, structurePrior = 1.0, maxDegree = 3, maxPathLength = -1, alpha = 0.05, completeRuleSetUsed = False, faithfulnessAssumed = True, verbose = False, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
        tetradData = pycausal.loadMixedData(df, numCategoriesToDiscretize)
        
        indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestConditionalGaussianLRT')(tetradData, alpha)
        
        score = javabridge.JClassWrapper('edu.cmu.tetrad.search.ConditionalGaussianScore')(tetradData, structurePrior, True)
        score.setPenaltyDiscount(penaltyDiscount) # set to 2 if variable# <= 50 otherwise set it to 4
        
        gfci = None
        
        if numBootstrap < 1:
            gfci = javabridge.JClassWrapper('edu.cmu.tetrad.search.GFci')(indTest, score)
            gfci.setMaxDegree(maxDegree)
            gfci.setMaxPathLength(maxPathLength)
            gfci.setCompleteRuleSetUsed(completeRuleSetUsed)
            gfci.setFaithfulnessAssumed(faithfulnessAssumed)
        else:
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.oracle.pag.Gfci')(indTest, score)
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('maxDegree', maxDegree)
            parameters.set('maxPathLength', maxPathLength)
            parameters.set('completeRuleSetUsed', completeRuleSetUsed)
            parameters.set('faithfulnessAssumed', faithfulnessAssumed)
            parameters.set('verbose', verbose)
            
            gfci = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble;')
            gfci.setEdgeEnsemble(edgeEnsemble)
            gfci.setParameters(parameters)
            
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
    
    def __init__(self, df, structurePrior = 1.0, samplePrior = 1.0, maxDegree = 3, maxPathLength = -1, alpha = 0.05, completeRuleSetUsed = False, faithfulnessAssumed = True, verbose = False, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
        tetradData = pycausal.loadDiscreteData(df)
        
        indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, alpha)
        
        score = javabridge.JClassWrapper('edu.cmu.tetrad.search.BDeuScore')(tetradData)
        score.setStructurePrior(structurePrior)
        score.setSamplePrior(samplePrior)
        
        gfci = None
        
        if numBootstrap < 1:
            gfci = javabridge.JClassWrapper('edu.cmu.tetrad.search.GFci')(indTest, score)
            gfci.setMaxDegree(maxDegree)
            gfci.setMaxPathLength(maxPathLength)
            gfci.setCompleteRuleSetUsed(completeRuleSetUsed)
            gfci.setFaithfulnessAssumed(faithfulnessAssumed)
        else:
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.oracle.pag.Gfci')(indTest, score)
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('maxDegree', maxDegree)
            parameters.set('maxPathLength', maxPathLength)
            parameters.set('completeRuleSetUsed', completeRuleSetUsed)
            parameters.set('faithfulnessAssumed', faithfulnessAssumed)
            parameters.set('verbose', verbose)
            
            gfci = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble;')
            gfci.setEdgeEnsemble(edgeEnsemble)
            gfci.setParameters(parameters)
            
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
    
    def __init__(self, df, penaltyDiscount = 4, maxDegree = 3, maxPathLength = -1, alpha = 0.05, completeRuleSetUsed = False, faithfulnessAssumed = True, verbose = False, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
        tetradData = pycausal.loadContinuousData(df)
        
        indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, alpha)
        
        score = javabridge.JClassWrapper('edu.cmu.tetrad.search.SemBicScore')(tetradData)
        score.setPenaltyDiscount(penaltyDiscount) # set to 2 if variable# <= 50 otherwise set it to 4
        
        gfci = None
        
        if numBootstrap < 1:
            gfci = javabridge.JClassWrapper('edu.cmu.tetrad.search.GFci')(indTest, score)
            gfci.setMaxDegree(maxDegree)
            gfci.setMaxPathLength(maxPathLength)
            gfci.setCompleteRuleSetUsed(completeRuleSetUsed)
            gfci.setFaithfulnessAssumed(faithfulnessAssumed)
        else:
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.oracle.pag.Gfci')(indTest, score)
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('maxDegree', maxDegree)
            parameters.set('maxPathLength', maxPathLength)
            parameters.set('completeRuleSetUsed', completeRuleSetUsed)
            parameters.set('faithfulnessAssumed', faithfulnessAssumed)
            parameters.set('verbose', verbose)
            
            gfci = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble;')
            gfci.setEdgeEnsemble(edgeEnsemble)
            gfci.setParameters(parameters)
            
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
    
    def __init__(self, df, dataType = 0, numCategoriesToDiscretize = 4, depth = 3, alpha = 0.05, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
        tetradData = None
        indTest = None
        
        # Continuous
        if dataType == 0:
            tetradData = pycausal.loadContinuousData(df)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, alpha)
        # Discrete
        elif dataType == 1:
            tetradData = pycausal.loadDiscreteData(df)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, alpha)
        # Mixed
        else:
            tetradData = pycausal.loadMixedData(df, numCategoriesToDiscretize)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestConditionalGaussianLRT')(tetradData, alpha)
        
        ccd = None
        
        if numBootstrap < 1:
            ccd = javabridge.JClassWrapper('edu.cmu.tetrad.search.Ccd')(indTest)
            ccd.setDepth(depth)
        else:
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.oracle.pag.Ccd')(indTest)
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('depth', depth)
            
            ccd = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble;')
            ccd.setEdgeEnsemble(edgeEnsemble)
            ccd.setParameters(parameters)
        
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
    
    def __init__(self, df, dataType = 0, numCategoriesToDiscretize = 4, depth = 3, alpha = 0.05, verbose = False, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
        tetradData = None
        indTest = None
        
        # Continuous
        if dataType == 0:
            tetradData = pycausal.loadContinuousData(df)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, alpha)
        # Discrete
        elif dataType == 1:
            tetradData = pycausal.loadDiscreteData(df)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, alpha)
        # Mixed
        else:
            tetradData = pycausal.loadMixedData(df, numCategoriesToDiscretize)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestConditionalGaussianLRT')(tetradData, alpha)
        
        pc = None
        
        if numBootstrap < 1:
            pc = javabridge.JClassWrapper('edu.cmu.tetrad.search.Pc')(indTest)
            pc.setDepth(depth)
        else:
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.oracle.pattern.Pc')(indTest)
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('depth', depth)
            parameters.set('verbose', verbose)
            
            pc = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble;')
            pc.setEdgeEnsemble(edgeEnsemble)
            pc.setParameters(parameters)
        
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
    
    def __init__(self, df, dataType = 0, numCategoriesToDiscretize = 4, depth = 3, maxPathLength = 3, useHeuristic = True, alpha = 0.05, verbose = False, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
        tetradData = None
        indTest = None
        
        # Continuous
        if dataType == 0:
            tetradData = pycausal.loadContinuousData(df)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, alpha)
        # Discrete
        elif dataType == 1:
            tetradData = pycausal.loadDiscreteData(df)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, alpha)
        # Mixed
        else:
            tetradData = pycausal.loadMixedData(df, numCategoriesToDiscretize)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestConditionalGaussianLRT')(tetradData, alpha)
        
        pcmax = None
        
        if numBootstrap < 1:
            pcmax = javabridge.JClassWrapper('edu.cmu.tetrad.search.PcStableMax')(indTest)
            pcmax.setDepth(depth)
            pcmax.setMaxPathLength(maxPathLength)
            pcmax.setUseHeuristic(useHeuristic)
        else:
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.oracle.pattern.PcStableMax')(indTest)
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('depth', depth)
            parameters.set('maxPathLength', maxPathLength)
            parameters.set('useMaxPOrientationHeuristic', useHeuristic)
            parameters.set('verbose', verbose)
            
            pcmax = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble;')
            pcmax.setEdgeEnsemble(edgeEnsemble)
            pcmax.setParameters(parameters)
            
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
    
    def __init__(self, df, dataType = 0, numCategoriesToDiscretize = 4, depth = 3, alpha = 0.05, verbose = False, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
        tetradData = None
        indTest = None
        
        # Continuous
        if dataType == 0:
            tetradData = pycausal.loadContinuousData(df)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, alpha)
        # Discrete
        elif dataType == 1:
            tetradData = pycausal.loadDiscreteData(df)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, alpha)
        # Mixed
        else:
            tetradData = pycausal.loadMixedData(df, numCategoriesToDiscretize)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestConditionalGaussianLRT')(tetradData, alpha)
        
        pcstable = None
        
        if numBootstrap < 1:
            pcstable = javabridge.JClassWrapper('edu.cmu.tetrad.search.PcStable')(indTest)
            pcstable.setDepth(depth)
            pcstable.setAggressivelyPreventCycles(aggressivelyPreventCycles)
        else:
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.oracle.pattern.PcStable')(indTest)
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('depth', depth)
            parameters.set('verbose', verbose)
            
            pcstable = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble;')
            pcstable.setEdgeEnsemble(edgeEnsemble)
            pcstable.setParameters(parameters)
        
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
    
    def __init__(self, df, dataType = 0, numCategoriesToDiscretize = 4, depth = 3, alpha = 0.05, verbose = False, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
        tetradData = None
        indTest = None
        
        # Continuous
        if dataType == 0:
            tetradData = pycausal.loadContinuousData(df)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, alpha)
        # Discrete
        elif dataType == 1:
            tetradData = pycausal.loadDiscreteData(df)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, alpha)
        # Mixed
        else:
            tetradData = pycausal.loadMixedData(df, numCategoriesToDiscretize)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestConditionalGaussianLRT')(tetradData, alpha)
        
        cpc = None
        
        if numBootstrap < 1:
            cpc = javabridge.JClassWrapper('edu.cmu.tetrad.search.Cpc')(indTest)
            cpc.setDepth(depth)
        else:
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.oracle.pattern.Cpc')(indTest)
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('depth', depth)
            parameters.set('verbose', verbose)
            
            cpc = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble;')
            cpc.setEdgeEnsemble(edgeEnsemble)
            cpc.setParameters(parameters)
            
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
    
    def __init__(self, df, dataType = 0, numCategoriesToDiscretize = 4, depth = 3, alpha = 0.05, verbose = False, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
        tetradData = None
        indTest = None
        
        # Continuous
        if dataType == 0:
            tetradData = pycausal.loadContinuousData(df)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, alpha)
        # Discrete
        elif dataType == 1:
            tetradData = pycausal.loadDiscreteData(df)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, alpha)
        # Mixed
        else:
            tetradData = pycausal.loadMixedData(df, numCategoriesToDiscretize)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestConditionalGaussianLRT')(tetradData, alpha)
        
        cpcstable = None
        
        if numBootstrap < 1:       
            cpcstable = javabridge.JClassWrapper('edu.cmu.tetrad.search.CpcStable')(indTest)
            cpcstable.setDepth(depth)
        else:
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.oracle.pattern.CpcStable')(indTest)
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('depth', depth)
            parameters.set('verbose', verbose)
            
            cpcstable = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu.pitt.dbmi.algo.bootstrap.BootstrapEdgeEnsemble;')
            cpcstable.setEdgeEnsemble(edgeEnsemble)
            cpcstable.setParameters(parameters)
            
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
    
    def __init__(self, df, depth = 3, alpha = 0.05, verbose = False, priorKnowledge = None):
        tetradData = pycausal.loadDiscreteData(df)
        indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, alpha)
        
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
    
