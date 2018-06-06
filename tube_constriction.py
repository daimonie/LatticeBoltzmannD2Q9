import numpy as np
import sys as sys
import time as time
from latticeboltzmann import latticeBoltzmann
from numba import jit
 
class tubeConstriction(latticeBoltzmann):
    """ D2Q9 LB Karman Vortex Sheet class.""" 
    def geometry(self, location, height):
        """Defines the geometry, boundaries and suchlike. In the base class, this is a pipe. """

        #circular object      
        radius = (self.nodes[1]-height)/2
        
        first_circle_obstruction = lambda x, y: (x-self.nodes[0]*location)**2 + y**2 < (radius)**2
        second_circle_obstruction = lambda x, y: (x-self.nodes[0]*location)**2 + (y-self.nodes[1])**2 < (radius)**2
     
        is_obstacle = lambda x,y: first_circle_obstruction(x, y) or second_circle_obstruction(x, y)


        self.obstacles = np.array([[is_obstacle(y,x) for x in range(self.nodes[1])] for y in range(self.nodes[0])])
 
         #sinusoidal flow
        self.flow = np.fromfunction(lambda dd, xx, yy: (1-dd) * self.velocity * (1 + 1e-4*np.sin(yy/self.nodes[1]*2*np.pi)), (2, self.nodes[0], self.nodes[1]))

        #initial distributions
        self.distributions = self.equilibrium( 1.0, self.flow) 