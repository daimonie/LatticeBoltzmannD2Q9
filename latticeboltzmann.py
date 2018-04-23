import numpy as np
import sys as sys
from time import time
from numba import jit

@jit
def numba_equilibrium(basisVelocity, nodes, weights, density, velocities):
	"""Calculates the equilibrium distributions"""

	projection   = 3.0*np.dot(basisVelocity, velocities.transpose(1,0,2))

	velocitySquare = (3.0/2.0)*(velocities[0]**2 + velocities[1]**2)

	equilibriumDistribution = np.zeros((9,nodes[0], nodes[1]))
	for i in range(9):
		equilibriumDistribution[i,:,:] = density*weights[i]*(1.0 + projection[i] + 0.5*projection[i]**2 - velocitySquare)

	return equilibriumDistribution


class latticeBoltzmann:
	""" D2Q9 LB base class."""
	def __init__(self, reynolds, nodesHorizontal, nodesVertical, width):
		#I prefer keeping track of this for benchmarking
		self.startTime = time()
		
		#grid dimensions
		self.length = 1.0;
		self.width = width;

		#characteristic velocity and time
		self.velocity = 0.1 * np.sqrt(1/3);
		self.time = self.length / self.velocity;

		#calculate the viscosity given the reynolds number
		self.reynolds = reynolds;
		self.viscosity = self.velocity * self.length / self.reynolds;

		#3.0 is because you divide by the sound speed squared
		#relaxationTime is used for streaming
		self.relaxationTime = 3.0 * self.viscosity + 0.5;

		#grid parameters
		self.nodes = (nodesHorizontal, nodesVertical);

	def time(self):
		"""Returns a string that tells you how much computing time was spend"""
		return "Process took %.3f seconds." % (time() - startTime)

	def sum_population(self, particles):
		""" sums the population """
		return np.sum(particles, axis=0)
	def equilibrium(self, density, velocities):
		"""Return equilibrum distributions"""
		return numba_equilibrium( self.basisVelocity, self.nodes, self.basisWeights, density, velocities)
	
	def initialise(self):
		""" Initialises the grid, weights and other such things """

		#Define the velocity basis, meaning the Q9 vectors
		self.basisVelocity = np.array([(x,y) for x in [0,-1,1] for y in [0, -1, 1]]);

		#Define the weights of each Q9 basis vector
		self.basisWeights = 1/36 * np.ones(9)
		self.basisWeights[np.asarray([np.linalg.norm(ci)<1.1 for ci in self.basisVelocity])] = 1/9
		self.basisWeights[0] = 4/9

		# No-slip boundaries are implemented by a bounce back boundary. This means that
		#	some lattice vectors point back to the point we want to involve.
		self.noSlip = [self.basisVelocity.tolist().index((-self.basisVelocity[i]).tolist()) for i in range(9)]
 
		# These are special cased velocities. 
		self.special = [
			np.arange(9)[np.asarray([ci[0] < 0 for ci in self.basisVelocity])],
			np.arange(9)[np.asarray([ci[0] == 0 for ci in self.basisVelocity])],
			np.arange(9)[np.asarray([ci[0] > 0 for ci in self.basisVelocity])],
		]



	def geometry(self):
		"""Defines the geometry, boundaries and suchlike. In the base class, this is a pipe. """

		self.obstacles = np.fromfunction(lambda xx, yy: 0, (self.nodes[0], self.nodes[1]));

		self.flow = np.fromfunction(lambda dd, xx, yy: (1-dd) * self.velocity * (1 + 1e-4*np.sin(yy/self.width*2*np.pi)), (2, self.nodes[0], self.nodes[1]));

		#initial distributions
		self.distributions = self.equilibrium( 1.0, self.flow);
