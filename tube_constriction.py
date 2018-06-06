import numpy as np
import sys as sys
import time as time
from latticeboltzmann import latticeBoltzmann
from numba import jit
 
class tubeConstriction(latticeBoltzmann):
    """ D2Q9 LB Karman Vortex Sheet class.""" 
    def geometry(self, location):
        """Defines the geometry, boundaries and suchlike. In the base class, this is a pipe. """

        if location > 1:
        	raise ValueError("karmanVortexSheet::geometry: Location %.3f should be a fractional number < 1." % location)
        #circular object
        self.obstacles = np.fromfunction(lambda x, y: ((x-self.nodes[0]*location)**2 + (y - self.nodes[1]/2)**2) < (self.length_scale)**2, (self.nodes[0], self.nodes[1]))
 
         #sinusoidal flow
        self.flow = np.fromfunction(lambda dd, xx, yy: (1-dd) * self.velocity * (1 + 1e-4*np.sin(yy/self.nodes[1]*2*np.pi)), (2, self.nodes[0], self.nodes[1]))

        #initial distributions
        self.distributions = self.equilibrium( 1.0, self.flow) 