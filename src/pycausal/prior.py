'''

Copyright (C) 2015 University of Pittsburgh.
 
This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.
 
This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.
 
You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
MA 02110-1301  USA
 
Created on July 8, 2016

@author: Chirayu Wongchokprasitti, PhD 
@email: chw20@pitt.edu
'''

import javabridge
import os
import glob

def knowledge(self, forbiddirect = None, requiredirect = None, addtemporal = None):
    tetrad_libdir = os.path.join(os.path.dirname(__file__), 'lib')

    for l in glob.glob(tetrad_libdir + os.sep + "*.jar"):
        javabridge.JARS.append(str(l))
            
    javabridge.start_vm(run_headless=True, max_heap_size = java_max_heap_size)
    javabridge.attach()        
    
    prior = javabridge.JClassWrapper('edu.cmu.tetrad.data.Knowledge2')()
    
    # forbidden directed edges
    if forbiddirect is not None:
        for i in range(0,len(forbiddirect)):
            forbid = forbiddirect[i]
            _from = forbid[0]
            _to = forbid[1]
            prior.setForbidden(_from, _to)
    
    # required directed edges
    if requiredirect is not None:
        for i in range(0,len(requiredirect)):
            require = requiredirect[i]
            _from = forbid[0]
            _to = forbid[1]
            prior.setRequired(_from, _to)
    
    # add temporal nodes' tiers
    if addtemporal is not None:
        for i in range(0,len(addtemporal)):
            teir = i
            temporal = addtemporal[i]
            for j in range(0,len(temporal)):
                node = temporal[j]
                node = node.replace(' ', '.')
                prior.addToTier(teir, node)
    
    return prior

def knowledgeFromFile(self, knowlegeFile):
    tetrad_libdir = os.path.join(os.path.dirname(__file__), 'lib')

    for l in glob.glob(tetrad_libdir + os.sep + "*.jar"):
        javabridge.JARS.append(str(l))
            
    javabridge.start_vm(run_headless=True, max_heap_size = java_max_heap_size)
    javabridge.attach()        
    
    f = javabridge.JClassWrapper('java.io.File')(knowlegeFile)
    reader = javabridge.JClassWrapper('edu.cmu.tetrad.data.DataReader')()
    prior = reader.parseKnowledge(f)

    return prior
    