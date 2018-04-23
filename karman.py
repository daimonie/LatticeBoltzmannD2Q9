import numpy as np
import sys as sys
import time as time
from latticeboltzmann import latticeBoltzmann
from numba import jit
 
class karmanVortexSheet(latticeBoltzmann):
	""" D2Q9 LB Karman Vortex Sheet class.""" 
	def geometry(self):
		"""Defines the geometry, boundaries and suchlike. In the base class, this is a pipe. """

		#circular object
		self.obstacles = np.fromfunction(lambda x, y: ((x-self.nodes[0]/4)**2 + (y - self.nodes[1]/2)**2) < (self.nodes[1]/9)**2, (self.nodes[0], self.nodes[1]))
 
 		#sinusoidal flow
		self.flow = np.fromfunction(lambda dd, xx, yy: (1-dd) * self.velocity * (1 + 1e-4*np.sin(yy/self.nodes[1]*2*np.pi)), (2, self.nodes[0], self.nodes[1]))

		#initial distributions
		self.distributions = self.equilibrium( 1.0, self.flow)
