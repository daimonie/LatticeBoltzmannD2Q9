import numpy as np
import sys as sys
import time as time
from latticeboltzmann import latticeBoltzmann
from numba import jit
 
class tubeConstriction(latticeBoltzmann):
    """ D2Q9 LB Karman Vortex Sheet class.""" 
    def geometry(self, location, width, height):
        """Defines the geometry, boundaries and suchlike. In the base class, this is a pipe. """

        #circular object
        
        left_horizontal = self.nodes[0]*location-width/2
        right_horizontal =  self.nodes[0]*location+width/2

        upper_vertical = self.nodes[1]/2-height/2
        lower_vertical = self.nodes[1]/2+height/2

        is_obstacle_horizontal = lambda x: (x > left_horizontal) and  (x < right_horizontal)
        is_obstacle_vertical = lambda y:  (y < upper_vertical) or (y > lower_vertical) 
        
        is_obstacle = lambda x, y: is_obstacle_horizontal(x) and is_obstacle_vertical(y)

        self.obstacles = np.array([[is_obstacle(y,x) for x in range(self.nodes[1])] for y in range(self.nodes[0])])
 
         #sinusoidal flow
        self.flow = np.fromfunction(lambda dd, xx, yy: (1-dd) * self.velocity * (1 + 1e-4*np.sin(yy/self.nodes[1]*2*np.pi)), (2, self.nodes[0], self.nodes[1]))

        #initial distributions
        self.distributions = self.equilibrium( 1.0, self.flow) 