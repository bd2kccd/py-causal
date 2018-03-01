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

import pycausal

class tetradrunner():
    
    algos = {}
    tests = {}
    scores = {}
    algoFactory = None
    paramDescs = None
    
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
                this.algos[str(algo.getAnnotation().command())] = algo
            
        testAnnotations = javabridge.JClassWrapper("edu.cmu.tetrad.annotation.TestOfIndependenceAnnotations")
        testClasses = testAnnotations.getInstance().getAnnotatedClasses()

        for i in range(0,testClasses.size()):
            test = testClasses.get(i)
            this.tests[str(test.getAnnotation().command())] = test

        scoreAnnotations = javabridge.JClassWrapper("edu.cmu.tetrad.annotation.ScoreAnnotations")
        scoreClasses = ScoreAnnotations.getInstance().getAnnotatedClasses()

        for i in range(0,scoreClasses.size()):
            score = scoreClasses.get(i)
            this.scores[str(score.getAnnotation().command())] = score
            
        this.algoFactory = javabridge.JClassWrapper("edu.cmu.tetrad.algcomparison.algorithm.AlgorithmFactory")
        paramDescs = javabridge.JClassWrapper("edu.cmu.tetrad.util.ParamDescriptions")
        this.paramDescs = paramDescs.getInstance()

    def run(self, algoId, dfs, testId = None, scoreId = None, priorKnowledge = None, dataType = 0, numCategoriesToDiscretize = 4, **parameters):
        algo = this.algos.get(algoId)
        algoAnno = algo.getAnnotation()
        algoClass = algo.getClazz()
        
        testClass = None
        if testId is not None:
            testClass = this.tests.get(testId)
            
        scoreClass = None
        if scoreId is not None:
            scoreClass = this.scores.get(scoreId)
        
        params = javabridge.JClassWrapper('edu.cmu.tetrad.util.Parameters')()
        for key in parameters.keys():
            if this.paramDescs.get(key) is not None:
                value = parameters[key]
                params.set(key, value)
                
        if priorKnowledge is not None:
            fges.setKnowledge(priorKnowledge)
    
        tetradData = None
        if isinstance(dfs, list):
            
            # Continuous
            if dataType == 0:
                tetradData = pycausal.loadContinuousData(dfs)
            # Discrete
            elif dataType == 1:
                tetradData = pycausal.loadDiscreteData(dfs)
            else:
                numCategoriesToDiscretize = 4
                if parameters['numCategoriesToDiscretize'] is not None:
                    numCategoriesToDiscretize = parameters['numCategoriesToDiscretize']
                tetradData = pycausal.loadMixedData(dfs, numCategoriesToDiscretize)
                
        else:
        
            tetradData = javabridge.JClassWrapper('java.util.ArrayList')()
            for df in dfs:
                dataset = None
                # Continuous
                if dataType == 0:
                    dataset = pycausal.loadContinuousData(df)
                # Discrete
                elif dataType == 1:
                    dataset = pycausal.loadDiscreteData(df)
                else:
                    dataset = pycausal.loadMixedData(df, numCategoriesToDiscretize)
                tetradData.add(dataset)
            
        algorithm = this.algoFactory.create(algoClass, testClass, scoreClass)
        
        if priorKnowledge is not None:
            algorithm.setKnowledge(priorKnowledge)
        
        self.tetradGraph = algorithm.search(tetradData, params)
        self.nodes = pycausal.extractTetradGraphNodes(self.tetradGraph)
        self.edges = pycausal.extractTetradGraphEdges(self.tetradGraph)
        self.graph = pycausal.generatePyDotGraph(self.nodes,self.edges)
        
    def listAlgorithms(self):
        _algos = this.algos.keys()
        _algos.sort()
        print('\n'.join(_algos))
    
    def listIndTests(self):
        _tests = this.tests.keys()
        _tests.sort()
        print('\n'.join(_tests))
    
    def listScores(self):
        _scores = this.scores.keys()
        _scores.sort()
        print('\n'.join(_scores))

    def getAlgorithmDescription(self, algoId):
        algo = this.algos.get(algoId)
        algoAnno = algo.getAnnotation()
        print(algoAnno.name() + ': ' + algoAnno.description())
    
    def getAlgorithmParameters(self, algoId, testId = None, scoreId = None):
        algo = this.algos.get(algoId)
        algoAnno = algo.getAnnotation()
        algoClass = algo.getClazz()
        
        testClass = None
        if testId is not None:
            testClass = this.tests.get(testId)
            
        scoreClass = None
        if scoreId is not None:
            scoreClass = this.scores.get(scoreId)
        
        algorithm = this.algoFactory.create(algoClass, testClass, scoreClass)
        algoParams = algorithm.getParameters()
  
        for i in range(0,algoParams.size()):
            algoParam = str(algoParams.get(i))
            paramDesc = this.paramDescs.get(algoParam)
            defaultValue = paramDesc.getDefaultValue()
            javaClass = pycausal.getJavaClass(defaultValue)
            desc = str(paramDesc.getDescription())
    
            print(algoParam + ": " + desc + ' (' + javaClass + ') [default:' + str(defaultValue) + ']')
        
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
                tetradData = pycausal.loadContinuousData(df, outputDataset = True)
            # Discrete
            elif dataType == 1:
                tetradData = pycausal.loadDiscreteData(df)
            # Mixed
            else:
                tetradData = pycausal.loadMixedData(df, numCategoriesToDiscretize)
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
    
