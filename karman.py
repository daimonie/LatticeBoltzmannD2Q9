import numpy as np
import sys as sys
import time as time
from latticeboltzmann import latticeBoltzmann
from numba import jit
 
class karmanVortexSheet(latticeBoltzmann):
	""" D2Q9 LB Karman Vortex Sheet class.""" 
	def geometry(self):
		"""Defines the geometry, boundaries and suchlike. In the base class, this is a pipe. """

		self.obstacles = np.fromfunction(lambda x, y: ((x-self.length/4)**2 + (y - self.width/2)**2) < (self.width/9)**2, (self.nodes[0], self.nodes[1]))


		print("is there an object?")
		for i in range(self.obstacles.shape[0]):
			for j in range(self.obstacles.shape[1]):
				if self.obstacles[i,j] != False:
					print(self.obstacles[i, j])
		sys.exit(0)


		self.flow = np.fromfunction(lambda dd, xx, yy: (1-dd) * self.velocity * (1 + 1e-4*np.sin(yy/self.width*2*np.pi)), (2, self.nodes[0], self.nodes[1]))

		#initial distributions
		self.distributions = self.equilibrium( 1.0, self.flow)
