import numpy as np
import sys as sys
from time import time
from numba import jit, jitclass, int32, float32

def latticeboltzmann(object):
	def __init__(self, reynolds, nodesHorizontal, nodesVertical):
		self.startTime = time()
		
		self.charLength = 1.0;
		self.charVelocity = 0.1 * np.sqrt(1/3);
		self.charTime = self.charLength / self.charVelocity;

		self.reynolds = reynolds;
		self.viscosity = self.charVelocity * self.charLength / self.reynolds;

		#3.0 is because you divide by the sound speed squared
		self.relaxationTime = 3.0 * self.viscosity + 0.5;

	def time(self):
		return "Process took %.3f seconds." % (time() - startTime)