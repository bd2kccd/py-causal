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

# rule: IGCI, R1TimeLag, R1, R2, R3, R4, Tanh, EB, Skew, SkewE, RSkew, RSkewE, Patel, Patel25, Patel50, Patel75, Patel90, FastICA, RC, Nlo
# score: andersonDarling, skew, kurtosis, fifthMoment, absoluteValue, exp, expUnstandardized, expUnstandardizedInverted, other, logcosh, entropy
class lofs():
    
    tetradGraph = None
    graph = None
    nodes = []
    edges = []
    
    def __init__(self, tetradGraph, dfs, dataType = 0, numCategoriesToDiscretize = 4, rule = 'R1', score = 'andersonDarling', alpha = 0.01, epsilon = 1.0, zeta = 0.0, orientStrongerDirection = False, r2Orient2Cycles = True, edgeCorrected = False, selfLoopStrength = 1.0):
        datasets = javabridge.JClassWrapper('java.util.ArrayList')()
        
        for idx in range(len(dfs)):
            df = dfs[idx]
            tetradData = None
            # Continuous
            if dataType == 0:
                tetradData = pycausal.loadContinuousData(df)
            # Discrete
            elif dataType == 1:
                tetradData = pycausal.loadDiscreteData(df)
            # Mixed
            else:
                tetradData = pycausal.loadMixedData(df, numCategoriesToDiscretize)
            datasets.add(tetradData)

        lofs2 = javabridge.JClassWrapper('edu.cmu.tetrad.search.Lofs2')(tetradGraph, datasets)
        rule = javabridge.get_static_field('edu.cmu.tetrad.search.Lofs2$Rule',
                                                   rule,
                                                   'Ledu/cmu/tetrad/search/Lofs2$Rule;')
        score = javabridge.get_static_field('edu.cmu.tetrad.search.Lofs$Score',
                                                   score,
                                                   'Ledu/cmu/tetrad/search/Lofs$Score;')
        lofs2.setRule(rule)
        lofs2.setScore(score)
        lofs2.setAlpha(alpha)
        lofs2.setEpsilon(epsilon)
        lofs2.setZeta(zeta)
        lofs2.setOrientStrongerDirection(orientStrongerDirection)
        lofs2.setR2Orient2Cycles(r2Orient2Cycles)
        lofs2.setEdgeCorrected(edgeCorrected)
        lofs2.setSelfLoopStrength(selfLoopStrength)
        
        self.tetradGraph = lofs2.orient()
        
        self.nodes = pycausal.extractTetradGraphNodes(self.tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(self.tetradGraph)
        self.graph = pycausal.generatePyDotGraph(self.nodes,self.edges)
        
    def getTetradGraph(self):
        return self.tetradGraph
    
    def getDot(self):
        return self.graph
    
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):
        return self.edges

class fofc():
    
    tetradGraph = None
    graph = None
    nodes = []
    edges = []
    
    def __init__(self, df, testType = 'TETRAD_WISHART', fofcAlgorithm = 'GAP', alpha = 0.01, numBootstrap = -1, ensembleMethod = 'Highest'):
        
        fofc = None
        
        if numBootstrap < 1:
            tetradData = pycausal.loadContinuousData(df)
            testType = javabridge.get_static_field('edu/cmu/tetrad/search/TestType',
                                                   testType,
                                                   'Ledu/cmu/tetrad/search/TestType;')
            fofcAlgorithm = javabridge.get_static_field('edu/cmu/tetrad/search/FindOneFactorClusters$Algorithm', 
                                                fofcAlgorithm, 
                                                'Ledu/cmu/tetrad/search/FindOneFactorClusters$Algorithm;')

            fofc = javabridge.JClassWrapper('edu.cmu.tetrad.search.FindOneFactorClusters')(tetradData, testType, fofcAlgorithm, alpha)
        else:
            tetradData = pycausal.loadContinuousData(df, outputDataset = True)
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
            edgeEnsemble = javabridge.get_static_field('edu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble;')
            fofc.setEdgeEnsemble(edgeEnsemble)
            fofc.setParameters(parameters)
            
        self.tetradGraph = fofc.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(self.tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(self.tetradGraph)
        self.graph = pycausal.generatePyDotGraph(self.nodes,self.edges)
        
    def getTetradGraph(self):
        return self.tetradGraph
    
    def getDot(self):
        return self.graph
    
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):
        return self.edges

class dm():
    
    tetradGraph = None
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
            
        self.tetradGraph = dm.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(self.tetradGraph, orig_columns, new_columns)
        self.edges = pycausal.extractTetradGraphEdges(self.tetradGraph, orig_columns, new_columns)
        self.graph = pycausal.generatePyDotGraph(self.nodes,self.edges)
        
    def getTetradGraph(self):
        return self.tetradGraph
    
    def getDot(self):
        return self.graph
    
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):
        return self.edges

class imagesBDeu():

    tetradGraph = None
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
            edgeEnsemble = javabridge.get_static_field('edu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble;')
            fges.setEdgeEnsemble(edgeEnsemble)
            fges.setParameters(parameters)

        fges.setVerbose(verbose)

        if priorKnowledge is not None:
            fges.setKnowledge(priorKnowledge)
    
        self.tetradGraph = fges.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(self.tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(self.tetradGraph)
        self.graph = pycausal.generatePyDotGraph(self.nodes,self.edges)
    
    def getTetradGraph(self):
        return self.tetradGraph
    
    def getDot(self):
        return self.graph
    
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):
        return self.edges

class imagesSemBic():

    tetradGraph = None
    graph = None
    nodes = []
    edges = []
    
    def __init__(self, dfs, penaltyDiscount = 4, maxDegree = 3, faithfulnessAssumed = True, verbose = False, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
        datasets = javabridge.JClassWrapper('java.util.ArrayList')()
        for idx in range(len(dfs)):
            df = dfs[idx]
            tetradData = None
            if numBootstrap < 1:
                tetradData = pycausal.loadContinuousData(df)
            else:
                tetradData = pycausal.loadContinuousData(df, outputDataset = True)
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
            edgeEnsemble = javabridge.get_static_field('edu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble;')
            fges.setEdgeEnsemble(edgeEnsemble)
            fges.setParameters(parameters)
        
        fges.setVerbose(verbose)
        
        if priorKnowledge is not None:
            fges.setKnowledge(priorKnowledge)
        
        self.tetradGraph = fges.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(self.tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(self.tetradGraph)
        self.graph = pycausal.generatePyDotGraph(self.nodes,self.edges)

    def getTetradGraph(self):
        return self.tetradGraph
    
    def getDot(self):
        return self.graph
    
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):
        return self.edges

class fas():
    
    tetradGraph = None
    nodes = []
    edges = []
    
    def __init__(self, df, dataType = 0, numCategoriesToDiscretize = 4, depth = 3, alpha = 0.05, verbose = False, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
        tetradData = None
        indTest = None
        
        # Continuous
        if dataType == 0:
            if numBootstrap < 1:                
                tetradData = pycausal.loadContinuousData(df)
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, alpha)
            else:
                tetradData = pycausal.loadContinuousData(df, outputDataset = True)
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.independence.FisherZ')()
        # Discrete
        elif dataType == 1:
            tetradData = pycausal.loadDiscreteData(df)
            if numBootstrap < 1:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, alpha)
            else:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.independence.ChiSquare')()
        # Mixed
        else:
            tetradData = pycausal.loadMixedData(df, numCategoriesToDiscretize)
            if numBootstrap < 1:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestConditionalGaussianLRT')(tetradData, alpha, False)
            else:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.independence.ConditionalGaussianLRT')()
                
        fas = None
        
        if numBootstrap < 1:
            fas = javabridge.JClassWrapper('edu.cmu.tetrad.search.Fas')(indTest)
            fas.setDepth(depth)
        else:
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.oracle.pattern.FAS')(indTest)
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('depth', depth)
            parameters.set('alpha', alpha)
            parameters.set('verbose', verbose)
            
            fas = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble;')
            fas.setEdgeEnsemble(edgeEnsemble)
            fas.setParameters(parameters)
            
        fas.setVerbose(verbose)
        
        if priorKnowledge is not None:
            fas.setKnowledge(priorKnowledge)
        
        self.tetradGraph = fas.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(self.tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(self.tetradGraph)

    def getTetradGraph(self):
        return self.tetradGraph
    
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):
        return self.edges

class fgesMixed():
    
    tetradGraph = None
    graph = None
    nodes = []
    edges = []
    
    def __init__(self, df, numCategoriesToDiscretize = 4, penaltyDiscount = 4, structurePrior = 1.0, maxDegree = 3, faithfulnessAssumed = True, verbose = False, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
        
        tetradData = pycausal.loadMixedData(df, numCategoriesToDiscretize)

        fges = None
        
        if numBootstrap < 1:
            score = javabridge.JClassWrapper('edu.cmu.tetrad.search.ConditionalGaussianScore')(tetradData, structurePrior, False)
            score.setPenaltyDiscount(penaltyDiscount) # set to 2 if variable# <= 50 otherwise set it to 4

            fges = javabridge.JClassWrapper('edu.cmu.tetrad.search.Fges')(score)
            fges.setMaxDegree(maxDegree)
            fges.setNumPatternsToStore(0)
            fges.setFaithfulnessAssumed(faithfulnessAssumed)
        else:
            score = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.score.ConditionalGaussianBicScore')()
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.oracle.pattern.Fges')(score)
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('penaltyDiscount', penaltyDiscount)
            parameters.set('structurePrior', structurePrior)
            parameters.set('maxDegree', maxDegree)
            parameters.set('faithfulnessAssumed', faithfulnessAssumed)
            parameters.set('verbose', verbose)
            
            fges = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble;')
            fges.setEdgeEnsemble(edgeEnsemble)
            fges.setParameters(parameters)
            
        fges.setVerbose(verbose)
        
        if priorKnowledge is not None:
            fges.setKnowledge(priorKnowledge)
        
        self.tetradGraph = fges.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(self.tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(self.tetradGraph)
        self.graph = pycausal.generatePyDotGraph(self.nodes,self.edges)

    def getTetradGraph(self):
        return self.tetradGraph
    
    def getDot(self):
        return self.graph
    
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):
        return self.edges

class fgesDiscrete():
    
    tetradGraph = None
    graph = None
    nodes = []
    edges = []
    
    def __init__(self, df, structurePrior = 1.0, samplePrior = 1.0, maxDegree = 3, faithfulnessAssumed = True, verbose = False, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
        
        tetradData = pycausal.loadDiscreteData(df)
        
        fges = None
        
        if numBootstrap < 1:
            score = javabridge.JClassWrapper('edu.cmu.tetrad.search.BDeuScore')(tetradData)
            score.setStructurePrior(structurePrior)
            score.setSamplePrior(samplePrior)

            fges = javabridge.JClassWrapper('edu.cmu.tetrad.search.Fges')(score)
            fges.setMaxDegree(maxDegree)
            fges.setNumPatternsToStore(0)
            fges.setFaithfulnessAssumed(faithfulnessAssumed)
        else:
            score = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.score.BdeuScore')()        
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.oracle.pattern.Fges')(score)
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('structurePrior', structurePrior)
            parameters.set('samplePrior', samplePrior)
            parameters.set('maxDegree', maxDegree)
            parameters.set('faithfulnessAssumed', faithfulnessAssumed)
            parameters.set('verbose', verbose)
            
            fges = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble;')
            fges.setEdgeEnsemble(edgeEnsemble)
            fges.setParameters(parameters)
            
        fges.setVerbose(verbose)
        
        if priorKnowledge is not None:    
            fges.setKnowledge(priorKnowledge)
            
        self.tetradGraph = fges.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(self.tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(self.tetradGraph)            
        self.graph = pycausal.generatePyDotGraph(self.nodes,self.edges)
        
        
    def getTetradGraph(self):
        return self.tetradGraph
    
    def getDot(self):
        return self.graph
    
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):
        return self.edges

    
class fges():
    
    tetradGraph = None
    graph = None
    nodes = []
    edges = []
    
    def __init__(self, df, penaltyDiscount = 4, maxDegree = 3, faithfulnessAssumed = True, verbose = False, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
            
        fges = None
        
        if numBootstrap < 1:
            tetradData = pycausal.loadContinuousData(df)

            score = javabridge.JClassWrapper('edu.cmu.tetrad.search.SemBicScore')(tetradData)
            score.setPenaltyDiscount(penaltyDiscount) # set to 2 if variable# <= 50 otherwise set it to 4
        
            fges = javabridge.JClassWrapper('edu.cmu.tetrad.search.Fges')(score)
            fges.setMaxDegree(maxDegree)
            fges.setNumPatternsToStore(0)
            fges.setFaithfulnessAssumed(faithfulnessAssumed)
        else:
            tetradData = pycausal.loadContinuousData(df, outputDataset = True)

            score = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.score.SemBicScore')()        
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.oracle.pattern.Fges')(score)
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('penaltyDiscount', penaltyDiscount)
            parameters.set('maxDegree', maxDegree)
            parameters.set('faithfulnessAssumed', faithfulnessAssumed)
            parameters.set('verbose', verbose)
            
            fges = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble;')
            fges.setEdgeEnsemble(edgeEnsemble)
            fges.setParameters(parameters)
            
        fges.setVerbose(verbose)
        
        if priorKnowledge is not None:    
            fges.setKnowledge(priorKnowledge)
            
        self.tetradGraph = fges.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(self.tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(self.tetradGraph)            
        self.graph = pycausal.generatePyDotGraph(self.nodes,self.edges)
        
    def getTetradGraph(self):
        return self.tetradGraph
    
    def getDot(self):
        return self.graph
    
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):    
        return self.edges

class fci():
    
    tetradGraph = None
    nodes = []
    edges = []
    
    def __init__(self, df, dataType = 0, numCategoriesToDiscretize = 4, depth = 3, alpha = 0.05, completeRuleSetUsed = False, verbose = False, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
        tetradData = None
        indTest = None
        
        # Continuous
        if dataType == 0:
            if numBootstrap < 1:                
                tetradData = pycausal.loadContinuousData(df)
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, alpha)
            else:
                tetradData = pycausal.loadContinuousData(df, outputDataset = True)
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.independence.FisherZ')()
        # Discrete
        elif dataType == 1:
            tetradData = pycausal.loadDiscreteData(df)
            if numBootstrap < 1:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, alpha)
            else:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.independence.ChiSquare')()
        # Mixed
        else:
            tetradData = pycausal.loadMixedData(df, numCategoriesToDiscretize)
            if numBootstrap < 1:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestConditionalGaussianLRT')(tetradData, alpha, False)
            else:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.independence.ConditionalGaussianLRT')()
            
        fci = None
        
        if numBootstrap < 1:
            fci = javabridge.JClassWrapper('edu.cmu.tetrad.search.Fci')(indTest)
            fci.setDepth(depth)
            fci.setCompleteRuleSetUsed(completeRuleSetUsed)
        else:
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.oracle.pag.Fci')(indTest)
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('depth', depth)
            parameters.set('alpha', alpha)
            parameters.set('completeRuleSetUsed', completeRuleSetUsed)
            parameters.set('verbose', verbose)
            
            fci = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble;')
            fci.setEdgeEnsemble(edgeEnsemble)
            fci.setParameters(parameters)
            
        fci.setVerbose(verbose)
        
        if priorKnowledge is not None:    
            fci.setKnowledge(priorKnowledge)
            
        self.tetradGraph = fci.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(self.tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(self.tetradGraph) 
        
    def getTetradGraph(self):
        return self.tetradGraph
    
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):    
        return self.edges
        
class cfci():
    
    tetradGraph = None
    nodes = []
    edges = []
    
    def __init__(self, df, dataType = 0, numCategoriesToDiscretize = 4, depth = 3, alpha = 0.05, verbose = False, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
        tetradData = None
        indTest = None
        
        # Continuous
        if dataType == 0:
            if numBootstrap < 1:                
                tetradData = pycausal.loadContinuousData(df)
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, alpha)
            else:
                tetradData = pycausal.loadContinuousData(df, outputDataset = True)
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.independence.FisherZ')()
        # Discrete
        elif dataType == 1:
            tetradData = pycausal.loadDiscreteData(df)
            if numBootstrap < 1:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, alpha)
            else:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.independence.ChiSquare')()
        # Mixed
        else:
            tetradData = pycausal.loadMixedData(df, numCategoriesToDiscretize)
            if numBootstrap < 1:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestConditionalGaussianLRT')(tetradData, alpha, False)
            else:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.independence.ConditionalGaussianLRT')()
        
        cfci = None
        
        if numBootstrap < 1:
            cfci = javabridge.JClassWrapper('edu.cmu.tetrad.search.Cfci')(indTest)
            cfci.setDepth(depth)
        else:
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.oracle.pag.Cfci')(indTest)
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('depth', depth)
            parameters.set('alpha', alpha)
            parameters.set('verbose', verbose)
            
            cfci = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble;')
            cfci.setEdgeEnsemble(edgeEnsemble)
            cfci.setParameters(parameters)
            
        cfci.setVerbose(verbose)
        
        if priorKnowledge is not None:    
            cfci.setKnowledge(priorKnowledge)
            
        self.tetradGraph = cfci.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(self.tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(self.tetradGraph) 
        
    def getTetradGraph(self):
        return self.tetradGraph
    
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):    
        return self.edges
        
class rfciMixed():
    
    tetradGraph = None
    nodes = []
    edges = []
    
    def __init__(self, df, numCategoriesToDiscretize = 4, depth = 3, maxPathLength = -1, alpha = 0.05, completeRuleSetUsed = False, verbose = False, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
        tetradData = pycausal.loadMixedData(df, numCategoriesToDiscretize)       
        
        rfci = None
        
        if numBootstrap < 1:
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestConditionalGaussianLRT')(tetradData, alpha, False)
            
            rfci = javabridge.JClassWrapper('edu.cmu.tetrad.search.Rfci')(indTest)
            rfci.setDepth(depth)
            rfci.setMaxPathLength(maxPathLength)
            rfci.setCompleteRuleSetUsed(completeRuleSetUsed)
        else:
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.independence.ConditionalGaussianLRT')()
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.oracle.pag.Rfci')(indTest)
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('depth', depth)
            parameters.set('maxPathLength', maxPathLength)
            parameters.set('alpha', alpha)
            parameters.set('completeRuleSetUsed', completeRuleSetUsed)
            parameters.set('verbose', verbose)
            
            rfci = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble;')
            rfci.setEdgeEnsemble(edgeEnsemble)
            rfci.setParameters(parameters)
            
        rfci.setVerbose(verbose)
        
        if priorKnowledge is not None:
            rfci.setKnowledge(priorKnowledge)
        
        self.tetradGraph = rfci.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(self.tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(self.tetradGraph)

    def getTetradGraph(self):
        return self.tetradGraph
    
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):
        return self.edges

class rfciDiscrete():
    
    tetradGraph = None
    nodes = []
    edges = []
    
    def __init__(self, df, depth = 3, maxPathLength = -1, alpha = 0.05, completeRuleSetUsed = False, verbose = False, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
        tetradData = pycausal.loadDiscreteData(df)
        
        rfci = None
        
        if numBootstrap < 1:
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, alpha)
            
            rfci = javabridge.JClassWrapper('edu.cmu.tetrad.search.Rfci')(indTest)
            rfci.setDepth(depth)
            rfci.setMaxPathLength(maxPathLength)
            rfci.setCompleteRuleSetUsed(completeRuleSetUsed)
        else:
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.independence.ChiSquare')()
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.oracle.pag.Rfci')(indTest)
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('depth', depth)
            parameters.set('maxPathLength', maxPathLength)
            parameters.set('alpha', alpha)
            parameters.set('completeRuleSetUsed', completeRuleSetUsed)
            parameters.set('verbose', verbose)
            
            rfci = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble;')
            rfci.setEdgeEnsemble(edgeEnsemble)
            rfci.setParameters(parameters)
        
        rfci.setVerbose(verbose)
        
        if priorKnowledge is not None:
            rfci.setKnowledge(priorKnowledge)
        
        self.tetradGraph = rfci.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(self.tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(self.tetradGraph)

    def getTetradGraph(self):
        return self.tetradGraph
    
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):
        return self.edges

class rfci():
    
    tetradGraph = None
    nodes = []
    edges = []
    
    def __init__(self, df, depth = 3, maxPathLength = -1, alpha = 0.05, completeRuleSetUsed = False, verbose = False, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
        
        rfci = None
        
        if numBootstrap < 1:
            tetradData = pycausal.loadContinuousData(df)
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, alpha)
        
            rfci = javabridge.JClassWrapper('edu.cmu.tetrad.search.Rfci')(indTest)
            rfci.setDepth(depth)
            rfci.setMaxPathLength(maxPathLength)
            rfci.setCompleteRuleSetUsed(completeRuleSetUsed)
        else:
            tetradData = pycausal.loadContinuousData(df, outputDataset = True)

            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.independence.FisherZ')()
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.oracle.pag.Rfci')(indTest)
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('depth', depth)
            parameters.set('maxPathLength', maxPathLength)
            parameters.set('alpha', alpha)
            parameters.set('completeRuleSetUsed', completeRuleSetUsed)
            parameters.set('verbose', verbose)
            
            rfci = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble;')
            rfci.setEdgeEnsemble(edgeEnsemble)
            rfci.setParameters(parameters)
            
        rfci.setVerbose(verbose)
        
        if priorKnowledge is not None:    
            rfci.setKnowledge(priorKnowledge)
            
        self.tetradGraph = rfci.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(self.tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(self.tetradGraph) 
        
    def getTetradGraph(self):
        return self.tetradGraph
    
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):    
        return self.edges
        
class gfciMixed():
    
    tetradGraph = None
    nodes = []
    edges = []
    
    def __init__(self, df, numCategoriesToDiscretize = 4, penaltyDiscount = 4, structurePrior = 1.0, maxDegree = 3, maxPathLength = -1, alpha = 0.05, completeRuleSetUsed = False, faithfulnessAssumed = True, verbose = False, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
        tetradData = pycausal.loadMixedData(df, numCategoriesToDiscretize)
        
        gfci = None
        
        if numBootstrap < 1:
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestConditionalGaussianLRT')(tetradData, alpha, False)

            score = javabridge.JClassWrapper('edu.cmu.tetrad.search.ConditionalGaussianScore')(tetradData, structurePrior, False)
            score.setPenaltyDiscount(penaltyDiscount) # set to 2 if variable# <= 50 otherwise set it to 4

            gfci = javabridge.JClassWrapper('edu.cmu.tetrad.search.GFci')(indTest, score)
            gfci.setMaxDegree(maxDegree)
            gfci.setMaxPathLength(maxPathLength)
            gfci.setCompleteRuleSetUsed(completeRuleSetUsed)
            gfci.setFaithfulnessAssumed(faithfulnessAssumed)
        else:
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.independence.ConditionalGaussianLRT')()
            score = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.score.ConditionalGaussianBicScore')()
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.oracle.pag.Gfci')(indTest, score)
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('penaltyDiscount', penaltyDiscount)
            parameters.set('structurePrior', structurePrior)
            parameters.set('maxDegree', maxDegree)
            parameters.set('maxPathLength', maxPathLength)
            parameters.set('alpha', alpha)
            parameters.set('completeRuleSetUsed', completeRuleSetUsed)
            parameters.set('faithfulnessAssumed', faithfulnessAssumed)
            parameters.set('verbose', verbose)
            
            gfci = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble;')
            gfci.setEdgeEnsemble(edgeEnsemble)
            gfci.setParameters(parameters)
            
        gfci.setVerbose(verbose)
        
        if priorKnowledge is not None:
            gfci.setKnowledge(priorKnowledge)
        
        self.tetradGraph = gfci.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(self.tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(self.tetradGraph)

    def getTetradGraph(self):
        return self.tetradGraph
    
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):
        return self.edges

class gfciDiscrete():
    
    tetradGraph = None
    nodes = []
    edges = []
    
    def __init__(self, df, structurePrior = 1.0, samplePrior = 1.0, maxDegree = 3, maxPathLength = -1, alpha = 0.05, completeRuleSetUsed = False, faithfulnessAssumed = True, verbose = False, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
        tetradData = pycausal.loadDiscreteData(df)
        
        gfci = None
        
        if numBootstrap < 1:
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, alpha)

            score = javabridge.JClassWrapper('edu.cmu.tetrad.search.BDeuScore')(tetradData)
            score.setStructurePrior(structurePrior)
            score.setSamplePrior(samplePrior)

            gfci = javabridge.JClassWrapper('edu.cmu.tetrad.search.GFci')(indTest, score)
            gfci.setMaxDegree(maxDegree)
            gfci.setMaxPathLength(maxPathLength)
            gfci.setCompleteRuleSetUsed(completeRuleSetUsed)
            gfci.setFaithfulnessAssumed(faithfulnessAssumed)
        else:
            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.independence.ChiSquare')()
            score = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.score.BdeuScore')()
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.oracle.pag.Gfci')(indTest, score)
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('structurePrior', structurePrior)
            parameters.set('samplePrior', samplePrior)
            parameters.set('maxDegree', maxDegree)
            parameters.set('maxPathLength', maxPathLength)
            parameters.set('alpha', alpha)
            parameters.set('completeRuleSetUsed', completeRuleSetUsed)
            parameters.set('faithfulnessAssumed', faithfulnessAssumed)
            parameters.set('verbose', verbose)
            
            gfci = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble;')
            gfci.setEdgeEnsemble(edgeEnsemble)
            gfci.setParameters(parameters)
            
        gfci.setVerbose(verbose)
        
        if priorKnowledge is not None:    
            gfci.setKnowledge(priorKnowledge)
            
        self.tetradGraph = gfci.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(self.tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(self.tetradGraph) 
        
    def getTetradGraph(self):
        return self.tetradGraph
    
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):    
        return self.edges        
        
class gfci():
    
    tetradGraph = None
    nodes = []
    edges = []
    
    def __init__(self, df, penaltyDiscount = 4, maxDegree = 3, maxPathLength = -1, alpha = 0.05, completeRuleSetUsed = False, faithfulnessAssumed = True, verbose = False, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):

        gfci = None
        
        if numBootstrap < 1:
            tetradData = pycausal.loadContinuousData(df)

            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, alpha)

            score = javabridge.JClassWrapper('edu.cmu.tetrad.search.SemBicScore')(tetradData)
            score.setPenaltyDiscount(penaltyDiscount) # set to 2 if variable# <= 50 otherwise set it to 4

            gfci = javabridge.JClassWrapper('edu.cmu.tetrad.search.GFci')(indTest, score)
            gfci.setMaxDegree(maxDegree)
            gfci.setMaxPathLength(maxPathLength)
            gfci.setCompleteRuleSetUsed(completeRuleSetUsed)
            gfci.setFaithfulnessAssumed(faithfulnessAssumed)
        else:
            tetradData = pycausal.loadContinuousData(df, outputDataset = True)

            indTest = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.independence.FisherZ')()
            score = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.score.SemBicScore')()
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.oracle.pag.Gfci')(indTest, score)
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('penaltyDiscount', penaltyDiscount)
            parameters.set('maxDegree', maxDegree)
            parameters.set('maxPathLength', maxPathLength)
            parameters.set('alpha', alpha)
            parameters.set('completeRuleSetUsed', completeRuleSetUsed)
            parameters.set('faithfulnessAssumed', faithfulnessAssumed)
            parameters.set('verbose', verbose)
            
            gfci = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble;')
            gfci.setEdgeEnsemble(edgeEnsemble)
            gfci.setParameters(parameters)
            
        gfci.setVerbose(verbose)
        
        if priorKnowledge is not None:    
            gfci.setKnowledge(priorKnowledge)
            
        self.tetradGraph = gfci.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(self.tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(self.tetradGraph) 
        
    def getTetradGraph(self):
        return self.tetradGraph
    
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):    
        return self.edges
        
class ccd():
    
    tetradGraph = None
    nodes = []
    edges = []
    
    def __init__(self, df, dataType = 0, numCategoriesToDiscretize = 4, depth = 3, alpha = 0.05, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
        tetradData = None
        indTest = None
        
        # Continuous
        if dataType == 0:
            if numBootstrap < 1:                
                tetradData = pycausal.loadContinuousData(df)
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, alpha)
            else:
                tetradData = pycausal.loadContinuousData(df, outputDataset = True)
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.independence.FisherZ')()
        # Discrete
        elif dataType == 1:
            tetradData = pycausal.loadDiscreteData(df)
            if numBootstrap < 1:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, alpha)
            else:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.independence.ChiSquare')()
        # Mixed
        else:
            tetradData = pycausal.loadMixedData(df, numCategoriesToDiscretize)
            if numBootstrap < 1:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestConditionalGaussianLRT')(tetradData, alpha, False)
            else:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.independence.ConditionalGaussianLRT')()
        
        ccd = None
        
        if numBootstrap < 1:
            ccd = javabridge.JClassWrapper('edu.cmu.tetrad.search.Ccd')(indTest)
            ccd.setDepth(depth)
        else:
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.oracle.pag.Ccd')(indTest)
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('depth', depth)
            parameters.set('alpha', alpha)
            
            ccd = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble;')
            ccd.setEdgeEnsemble(edgeEnsemble)
            ccd.setParameters(parameters)
        
        if priorKnowledge is not None:    
            ccd.setKnowledge(priorKnowledge)
            
        self.tetradGraph = ccd.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(self.tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(self.tetradGraph) 
        
    def getTetradGraph(self):
        return self.tetradGraph
    
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):    
        return self.edges
    
class pc():
    
    tetradGraph = None
    graph = None
    nodes = []
    edges = []
    
    def __init__(self, df, dataType = 0, numCategoriesToDiscretize = 4, depth = 3, alpha = 0.05, verbose = False, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
        tetradData = None
        indTest = None
        
        # Continuous
        if dataType == 0:
            if numBootstrap < 1:                
                tetradData = pycausal.loadContinuousData(df)
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, alpha)
            else:
                tetradData = pycausal.loadContinuousData(df, outputDataset = True)
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.independence.FisherZ')()
        # Discrete
        elif dataType == 1:
            tetradData = pycausal.loadDiscreteData(df)
            if numBootstrap < 1:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, alpha)
            else:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.independence.ChiSquare')()
        # Mixed
        else:
            tetradData = pycausal.loadMixedData(df, numCategoriesToDiscretize)
            if numBootstrap < 1:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestConditionalGaussianLRT')(tetradData, alpha, False)
            else:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.independence.ConditionalGaussianLRT')()
        
        pc = None
        
        if numBootstrap < 1:
            pc = javabridge.JClassWrapper('edu.cmu.tetrad.search.Pc')(indTest)
            pc.setDepth(depth)
        else:
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.oracle.pattern.Pc')(indTest)
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('depth', depth)
            parameters.set('alpha', alpha)
            parameters.set('verbose', verbose)
            
            pc = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble;')
            pc.setEdgeEnsemble(edgeEnsemble)
            pc.setParameters(parameters)
        
        pc.setVerbose(verbose)    
        
        if priorKnowledge is not None:    
            pc.setKnowledge(priorKnowledge)
            
        self.tetradGraph = pc.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(self.tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(self.tetradGraph)
        self.graph = pycausal.generatePyDotGraph(self.nodes,self.edges)

    def getTetradGraph(self):
        return self.tetradGraph
    
    def getDot(self):
        return self.graph
        
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):    
        return self.edges
    
class pcstablemax():
    
    tetradGraph = None
    graph = None
    nodes = []
    edges = []
    
    def __init__(self, df, dataType = 0, numCategoriesToDiscretize = 4, depth = 3, maxPathLength = 3, useHeuristic = True, alpha = 0.05, verbose = False, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
        tetradData = None
        indTest = None
        
        # Continuous
        if dataType == 0:
            if numBootstrap < 1:                
                tetradData = pycausal.loadContinuousData(df)
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, alpha)
            else:
                tetradData = pycausal.loadContinuousData(df, outputDataset = True)
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.independence.FisherZ')()
        # Discrete
        elif dataType == 1:
            tetradData = pycausal.loadDiscreteData(df)
            if numBootstrap < 1:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, alpha)
            else:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.independence.ChiSquare')()
        # Mixed
        else:
            tetradData = pycausal.loadMixedData(df, numCategoriesToDiscretize)
            if numBootstrap < 1:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestConditionalGaussianLRT')(tetradData, alpha, False)
            else:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.independence.ConditionalGaussianLRT')()
        
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
            parameters.set('alpha', alpha)
            parameters.set('verbose', verbose)
            
            pcmax = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble;')
            pcmax.setEdgeEnsemble(edgeEnsemble)
            pcmax.setParameters(parameters)
            
        pcmax.setVerbose(verbose)
        
        if priorKnowledge is not None:    
            pcmax.setKnowledge(priorKnowledge)
            
        self.tetradGraph = pcmax.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(self.tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(self.tetradGraph)
        self.graph = pycausal.generatePyDotGraph(self.nodes,self.edges)

    def getTetradGraph(self):
        return self.tetradGraph
    
    def getDot(self):
        return self.graph
        
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):    
        return self.edges
    
class pcstable():
    
    tetradGraph = None
    graph = None
    nodes = []
    edges = []
    
    def __init__(self, df, dataType = 0, numCategoriesToDiscretize = 4, depth = 3, alpha = 0.05, verbose = False, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
        tetradData = None
        indTest = None
        
        # Continuous
        if dataType == 0:
            if numBootstrap < 1:                
                tetradData = pycausal.loadContinuousData(df)
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, alpha)
            else:
                tetradData = pycausal.loadContinuousData(df, outputDataset = True)
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.independence.FisherZ')()
        # Discrete
        elif dataType == 1:
            tetradData = pycausal.loadDiscreteData(df)
            if numBootstrap < 1:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, alpha)
            else:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.independence.ChiSquare')()
        # Mixed
        else:
            tetradData = pycausal.loadMixedData(df, numCategoriesToDiscretize)
            if numBootstrap < 1:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestConditionalGaussianLRT')(tetradData, alpha, False)
            else:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.independence.ConditionalGaussianLRT')()
        
        pcstable = None
        
        if numBootstrap < 1:
            pcstable = javabridge.JClassWrapper('edu.cmu.tetrad.search.PcStable')(indTest)
            pcstable.setDepth(depth)
        else:
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.oracle.pattern.PcStable')(indTest)
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('depth', depth)
            parameters.set('alpha', alpha)
            parameters.set('verbose', verbose)
            
            pcstable = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble;')
            pcstable.setEdgeEnsemble(edgeEnsemble)
            pcstable.setParameters(parameters)
        
        pcstable.setVerbose(verbose)
        
        if priorKnowledge is not None:    
            pcstable.setKnowledge(priorKnowledge)
            
        self.tetradGraph = pcstable.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(self.tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(self.tetradGraph)
        self.graph = pycausal.generatePyDotGraph(self.nodes,self.edges)

    def getTetradGraph(self):
        return self.tetradGraph
    
    def getDot(self):
        return self.graph
        
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):    
        return self.edges    

class cpc():
    
    tetradGraph = None
    graph = None
    nodes = []
    edges = []
    
    def __init__(self, df, dataType = 0, numCategoriesToDiscretize = 4, depth = 3, alpha = 0.05, verbose = False, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
        tetradData = None
        indTest = None
        
        # Continuous
        if dataType == 0:
            if numBootstrap < 1:                
                tetradData = pycausal.loadContinuousData(df)
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, alpha)
            else:
                tetradData = pycausal.loadContinuousData(df, outputDataset = True)
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.independence.FisherZ')()
        # Discrete
        elif dataType == 1:
            tetradData = pycausal.loadDiscreteData(df)
            if numBootstrap < 1:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, alpha)
            else:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.independence.ChiSquare')()
        # Mixed
        else:
            tetradData = pycausal.loadMixedData(df, numCategoriesToDiscretize)
            if numBootstrap < 1:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestConditionalGaussianLRT')(tetradData, alpha, False)
            else:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.independence.ConditionalGaussianLRT')()
        
        cpc = None
        
        if numBootstrap < 1:
            cpc = javabridge.JClassWrapper('edu.cmu.tetrad.search.Cpc')(indTest)
            cpc.setDepth(depth)
        else:
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.oracle.pattern.Cpc')(indTest)
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('depth', depth)
            parameters.set('alpha', alpha)
            parameters.set('verbose', verbose)
            
            cpc = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble;')
            cpc.setEdgeEnsemble(edgeEnsemble)
            cpc.setParameters(parameters)
            
        cpc.setVerbose(verbose)
        
        if priorKnowledge is not None:    
            cpc.setKnowledge(priorKnowledge)
            
        self.tetradGraph = cpc.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(self.tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(self.tetradGraph)
        self.graph = pycausal.generatePyDotGraph(self.nodes,self.edges)

    def getTetradGraph(self):
        return self.tetradGraph
    
    def getDot(self):
        return self.graph
        
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):    
        return self.edges    
    
class cpcstable():
    
    tetradGraph = None
    graph = None
    nodes = []
    edges = []
    
    def __init__(self, df, dataType = 0, numCategoriesToDiscretize = 4, depth = 3, alpha = 0.05, verbose = False, priorKnowledge = None, numBootstrap = -1, ensembleMethod = 'Highest'):
        tetradData = None
        indTest = None
        
        # Continuous
        if dataType == 0:
            if numBootstrap < 1:                
                tetradData = pycausal.loadContinuousData(df)
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, alpha)
            else:
                tetradData = pycausal.loadContinuousData(df, outputDataset = True)
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.independence.FisherZ')()
        # Discrete
        elif dataType == 1:
            tetradData = pycausal.loadDiscreteData(df)
            if numBootstrap < 1:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, alpha)
            else:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.independence.ChiSquare')()
        # Mixed
        else:
            tetradData = pycausal.loadMixedData(df, numCategoriesToDiscretize)
            if numBootstrap < 1:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestConditionalGaussianLRT')(tetradData, alpha, False)
            else:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.independence.ConditionalGaussianLRT')()
        
        cpcstable = None
        
        if numBootstrap < 1:       
            cpcstable = javabridge.JClassWrapper('edu.cmu.tetrad.search.CpcStable')(indTest)
            cpcstable.setDepth(depth)
        else:
            algorithm = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.algorithm.oracle.pattern.CpcStable')(indTest)
            
            parameters = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
            parameters.set('depth', depth)
            parameters.set('alpha', alpha)
            parameters.set('verbose', verbose)
            
            cpcstable = javabridge.JClassWrapper('edu.pitt.dbmi.algo.bootstrap.GeneralBootstrapTest')(tetradData, algorithm, numBootstrap)
            edgeEnsemble = javabridge.get_static_field('edu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble',
                                               ensembleMethod,
                                               'Ledu/pitt/dbmi/algo/bootstrap/BootstrapEdgeEnsemble;')
            cpcstable.setEdgeEnsemble(edgeEnsemble)
            cpcstable.setParameters(parameters)
            
        cpcstable.setVerbose(verbose)
        
        if priorKnowledge is not None:    
            cpcstable.setKnowledge(priorKnowledge)
            
        self.tetradGraph = cpcstable.search()
        
        self.nodes = pycausal.extractTetradGraphNodes(self.tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(self.tetradGraph)
        self.graph = pycausal.generatePyDotGraph(self.nodes,self.edges)

    def getTetradGraph(self):
        return self.tetradGraph
    
    def getDot(self):
        return self.graph
        
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):    
        return self.edges    
    
class bayesEst():
    
    tetradGraph = None
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
            
        self.tetradGraph = cpc.search()
        dags = javabridge.JClassWrapper('edu.cmu.tetrad.search.DagInPatternIterator')(self.tetradGraph)
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
        
    def getTetradGraph(self):
        return self.tetradGraph
    
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
    
    tetradGraph = None
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
            
        self.tetradGraph = graph    
        self.nodes = pycausal.extractTetradGraphNodes(dag)
        self.edges = pycausal.extractTetradGraphEdges(dag)
        self.graph = pycausal.generatePyDotGraph(self.nodes,self.edges)
        self.dag = dag
        
    def getTetradGraph(self):
        return self.tetradGraph
    
    def getDot(self):
        return self.graph
        
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):    
        return self.edges    
    
    def getDag(self):
        return self.dag
    
