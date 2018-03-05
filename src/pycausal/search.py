'''
Created on Feb 17, 2016
Updated on Feb 28, 2018

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

import pycausal as pc

class tetradrunner():
    
    algos = {}
    tests = {}
    scores = {}
    paramDescs = None
    algoFactory = None
    
    tetradGraph = None
    graph = None
    nodes = []
    edges = []
    
    def __init__(self):
        algorithmAnnotations = javabridge.JClassWrapper("edu.cmu.tetrad.annotation.AlgorithmAnnotations")
        algoClasses = algorithmAnnotations.getInstance().getAnnotatedClasses()

        for i in range(0,algoClasses.size()):
            algo = algoClasses.get(i)
            algoType = str(algo.getAnnotation().algoType())
            if algoType != 'orient_pairwise':
                self.algos[str(algo.getAnnotation().command())] = algo
            
        testAnnotations = javabridge.JClassWrapper("edu.cmu.tetrad.annotation.TestOfIndependenceAnnotations")
        testClasses = testAnnotations.getInstance().getAnnotatedClasses()

        for i in range(0,testClasses.size()):
            test = testClasses.get(i)
            self.tests[str(test.getAnnotation().command())] = test

        scoreAnnotations = javabridge.JClassWrapper("edu.cmu.tetrad.annotation.ScoreAnnotations")
        scoreClasses = scoreAnnotations.getInstance().getAnnotatedClasses()

        for i in range(0,scoreClasses.size()):
            score = scoreClasses.get(i)
            self.scores[str(score.getAnnotation().command())] = score
            
        paramDescs = javabridge.JClassWrapper("edu.cmu.tetrad.util.ParamDescriptions")
        self.paramDescs = paramDescs.getInstance()

        self.algoFactory = javabridge.JClassWrapper("edu.cmu.tetrad.algcomparison.algorithm.AlgorithmFactory")
        
    def listAlgorithms(self):
        _algos = self.algos.keys()
        _algos.sort()
        print('\n'.join(_algos))
    
    def listIndTests(self):
        _tests = self.tests.keys()
        _tests.sort()
        print('\n'.join(_tests))
    
    def listScores(self):
        _scores = self.scores.keys()
        _scores.sort()
        print('\n'.join(_scores))

    def getAlgorithmDescription(self, algoId):
        algo = self.algos.get(algoId)
        algoAnno = algo.getAnnotation()
        print(algoAnno.name() + ': ' + algoAnno.description())
    
    def getAlgorithmParameters(self, algoId, testId = None, scoreId = None):
        algo = self.algos.get(algoId)
        algoClass = algo.getClazz()
        
        testClass = None
        if testId is not None:
            test = self.tests.get(testId)
            testClass = test.getClazz()
            
        scoreClass = None
        if scoreId is not None:
            score = self.scores.get(scoreId)
            scoreClass = score.getClazz()
        
        algorithm = self.algoFactory.create(algoClass, testClass, scoreClass)
        algoParams = algorithm.getParameters()
  
        for i in range(0,algoParams.size()):
            algoParam = str(algoParams.get(i))
            paramDesc = self.paramDescs.get(algoParam)
            defaultValue = paramDesc.getDefaultValue()
            javaClass = str(javabridge.call(javabridge.call(defaultValue.o, "getClass", "()Ljava/lang/Class;"),
                            "getName","()Ljava/lang/String;"))
            desc = str(paramDesc.getDescription())
    
            print(algoParam + ": " + desc + ' (' + javaClass + ') [default:' + str(defaultValue) + ']')
        
    def run(self, algoId, dfs, testId = None, scoreId = None, priorKnowledge = None, dataType = 0, numCategoriesToDiscretize = 4, **parameters):
        algo = self.algos.get(algoId)
        algoAnno = algo.getAnnotation()
        algoClass = algo.getClazz()
        
        testClass = None
        if testId is not None:
            test = self.tests.get(testId)
            testClass = test.getClazz()
            
        scoreClass = None
        if scoreId is not None:
            score = self.scores.get(scoreId)
            scoreClass = score.getClazz()
        
        params = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
        for key in parameters.keys():
            if self.paramDescs.get(key) is not None:
                value = parameters[key]
                params.set(key, value)
                
        tetradData = None
        if not isinstance(dfs, list):
            
            # Continuous
            if dataType == 0:
                if 'bootstrapSampleSize' in parameters and parameters['bootstrapSampleSize'] > 0:
                    tetradData = pc.loadContinuousData(dfs, outputDataset = True)
                else:
                    tetradData = pc.loadContinuousData(dfs)
            # Discrete
            elif dataType == 1:
                tetradData = pc.loadDiscreteData(dfs)
            else:
                tetradData = pc.loadMixedData(dfs, numCategoriesToDiscretize)
                
        else:
        
            tetradData = javabridge.JClassWrapper('java.util.ArrayList')()
            for df in dfs:
                dataset = None
                # Continuous
                if dataType == 0:
                    if 'bootstrapSampleSize' in parameters and parameters['bootstrapSampleSize'] > 0:
                        dataset = pc.loadContinuousData(dfs, outputDataset = True)
                    else:
                        dataset = pc.loadContinuousData(dfs)
                # Discrete
                elif dataType == 1:
                    dataset = pc.loadDiscreteData(df)
                else:
                    dataset = pc.loadMixedData(df, numCategoriesToDiscretize)
                tetradData.add(dataset)
            
        algorithm = self.algoFactory.create(algoClass, testClass, scoreClass)
        
        if priorKnowledge is not None:
            algorithm.setKnowledge(priorKnowledge)
        
        self.tetradGraph = algorithm.search(tetradData, params)
        self.nodes = pc.extractTetradGraphNodes(self.tetradGraph)
        self.edges = pc.extractTetradGraphEdges(self.tetradGraph)
        self.graph = pc.generatePyDotGraph(self.nodes,self.edges,self.tetradGraph)
        
    def getTetradGraph(self):
        return self.tetradGraph
    
    def getDot(self):
        return self.graph
    
    def getNodes(self):
        return self.nodes
    
    def getEdges(self):
        return self.edges

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
                tetradData = pc.loadContinuousData(df, outputDataset = True)
            # Discrete
            elif dataType == 1:
                tetradData = pc.loadDiscreteData(df)
            # Mixed
            else:
                tetradData = pc.loadMixedData(df, numCategoriesToDiscretize)
            datasets.add(tetradData)

        lofs2 = javabridge.JClassWrapper('edu.cmu.tetrad.search.Lofs2')(tetradGraph, datasets)
        rule = javabridge.get_static_field('edu/cmu/tetrad/search/Lofs2$Rule',
                                                   rule,
                                                   'Ledu/cmu/tetrad/search/Lofs2$Rule;')
        score = javabridge.get_static_field('edu/cmu/tetrad/search/Lofs$Score',
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
        
        self.nodes = pc.extractTetradGraphNodes(self.tetradGraph)
        self.edges = pc.extractTetradGraphEdges(self.tetradGraph)
        self.graph = pc.generatePyDotGraph(self.nodes,self.edges)
        
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
        
        tetradData = pc.loadContinuousData(df, outputDataset = True)
        
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
        
        self.nodes = pc.extractTetradGraphNodes(self.tetradGraph, orig_columns, new_columns)
        self.edges = pc.extractTetradGraphEdges(self.tetradGraph, orig_columns, new_columns)
        self.graph = pc.generatePyDotGraph(self.nodes,self.edges)
        
    def getTetradGraph(self):
        return self.tetradGraph
    
    def getDot(self):
        return self.graph
    
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
                tetradData = pc.loadContinuousData(df)
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestFisherZ')(tetradData, alpha)
            else:
                tetradData = pc.loadContinuousData(df, outputDataset = True)
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.independence.FisherZ')()
        # Discrete
        elif dataType == 1:
            tetradData = pc.loadDiscreteData(df)
            if numBootstrap < 1:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.search.IndTestChiSquare')(tetradData, alpha)
            else:
                indTest = javabridge.JClassWrapper('edu.cmu.tetrad.algcomparison.independence.ChiSquare')()
        # Mixed
        else:
            tetradData = pc.loadMixedData(df, numCategoriesToDiscretize)
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
    
class bayesEst():
    
    tetradGraph = None
    graph = None
    nodes = []
    edges = []
    dag = None
    bayesPm = None
    bayesIm = None
    
    def __init__(self, df, depth = 3, alpha = 0.05, verbose = False, priorKnowledge = None):
        tetradData = pc.loadDiscreteData(df)
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

        self.nodes = pc.extractTetradGraphNodes(dag)
        self.edges = pc.extractTetradGraphEdges(dag)
        self.graph = pc.generatePyDotGraph(self.nodes,self.edges)
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
        self.nodes = pc.extractTetradGraphNodes(dag)
        self.edges = pc.extractTetradGraphEdges(dag)
        self.graph = pc.generatePyDotGraph(self.nodes,self.edges)
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
    
