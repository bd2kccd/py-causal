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
 
Created on Feb 15, 2016

@author: Chirayu Wongchokprasitti, PhD 
@email: chw20@pitt.edu
'''

# lgpl 2.1
__author__ = 'Chirayu Kong Wongchokprasitti'
__version__ = '0.1.1'
__license__ = 'LGPL >= 2.1'


import javabridge
import os
import glob

def start_vm(self, java_max_heap_size = None):
    tetrad_libdir = os.path.join(os.path.dirname(__file__), 'lib')

    for l in glob.glob(tetrad_libdir + os.sep + "*.jar"):
        javabridge.JARS.append(str(l))
            
    javabridge.start_vm(run_headless=True, max_heap_size = java_max_heap_size)
    javabridge.attach()        
    
def stop_vm(self):
    javabridge.detach()
    javabridge.kill_vm()

def isNodeExisting(nodes,node):
    try:
        nodes.index(node)
        return True
    except IndexError:
        print "Node %s does not exist!", node
        return False
    
