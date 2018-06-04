import numpy as np
import sys as sys
import time as time
from numba import jit
import itertools #required by np.roll

@jit(parallel=True)
def numba_equilibrium(basis_velocity, nodes, weights, density, velocities):
    """Calculates the equilibrium distributions"""

    projection   = 3.0*np.dot(basis_velocity, velocities.transpose(1,0,2))

    velocity_square = (3.0/2.0)*(velocities[0]**2 + velocities[1]**2)

    equilibrium_distribution = np.zeros((9,nodes[0], nodes[1]))
    for i in range(9):
        equilibrium_distribution[i,:,:] = density*weights[i]*(1.0 + projection[i] + 0.5*projection[i]**2 - velocity_square)

    return equilibrium_distribution 

@jit(parallel=True)
def numba_collide(distributions, relaxation_time, equilibrium):
    return distributions - 1/relaxation_time * (distributions - equilibrium)
  
def numba_numpy_roll(a, shift):
    """
    Roll array elements along a given axis. 

    I just copy pasted it from the numpy implementation to use JIT/parallel. Removed a number of safety checks as well.

    """  

    #hard coding these
    axis = (0, 1)
    shifts = {0: 0, 1: 0}


    broadcasted = np.broadcast(shift, axis) 

    for sh, ax in broadcasted:
        shifts[ax] += sh

    rolls = [((slice(None), slice(None)),)] * a.ndim
    for ax, offset in shifts.items():
        offset %= a.shape[ax] or 1  # If `a` is empty, nothing matters.
        if offset:
            # (original, result), (original, result)
            rolls[ax] = ((slice(None, -offset), slice(offset, None)),
                         (slice(-offset, None), slice(None, offset)))

    result = np.empty_like(a)
    for indices in itertools.product(*rolls):
        arr_index, res_index = zip(*indices)
        result[res_index] = a[arr_index]

    return result
 
@jit(parallel=True)
def numba_roll( new_distributions, velocities):
    #implementation of numpy roll with JIT parallel

    shape = new_distributions.shape
    final_distributions = np.zeros(shape)  

    for i in range(shape[0]):
            final_distributions[i,:,:] = numba_numpy_roll(new_distributions[i,:,:], velocities[i, :])

    return final_distributions
class latticeBoltzmann:
    """ D2Q9 LB base class."""
    def __init__(self, reynolds, nodes_horizontal, nodes_vertical, length_scale):
        #I prefer keeping track of this for benchmarking
        self.start_time = time.time() 

        #characteristic velocity and time
        self.velocity = 0.1 * np.sqrt(1/3)
        self.time = nodes_horizontal / self.velocity

        #calculate the viscosity given the reynolds number
        self.reynolds = reynolds
        self.length_scale = length_scale
        self.viscosity = self.velocity * length_scale / self.reynolds

        #3.0 is because you divide by the sound speed squared
        #relaxation_time is used for streaming
        self.relaxation_time = 3.0 * self.viscosity + 0.5

        #grid parameters
        self.nodes = (nodes_horizontal, nodes_vertical) 
    def reset_time(self):
        self.start_time = time.time()
    def report_time(self):
        """Returns a string that tells you how much computing time was spend"""
        return (time.time() - self.start_time)

    def sum_population(self, particles):
        """ sums the population """
        return np.sum(particles, axis=0)
    def equilibrium(self, density, velocities):
        """Return equilibrum distributions"""
        return numba_equilibrium( self.basis_velocity, self.nodes, self.basis_weights, density, velocities)
    
    def initialise(self):
        """ Initialises the grid, weights and other such things """

        #Define the velocity basis, meaning the Q9 vectors
        self.basis_velocity = np.array([(x,y) for x in [0, -1, 1] for y in [0, -1, 1]])

        #Define the weights of each Q9 basis vector
        self.basis_weights = 1/36 * np.ones(9)
        self.basis_weights[np.asarray([np.linalg.norm(ci)<1.1 for ci in self.basis_velocity])] = 1/9
        self.basis_weights[0] = 4/9

        # No-slip boundaries are implemented by a bounce back boundary. This means that
        #    some lattice vectors point back to the point we want to involve.
        self.noSlip = [self.basis_velocity.tolist().index((-self.basis_velocity[i]).tolist()) for i in range(9)]
 
        # These are special cased velocities. 
        self.special = [
            np.arange(9)[np.asarray([ci[0] < 0 for ci in self.basis_velocity])],
            np.arange(9)[np.asarray([ci[0] == 0 for ci in self.basis_velocity])],
            np.arange(9)[np.asarray([ci[0] > 0 for ci in self.basis_velocity])],
        ]



    def geometry(self):
        """Defines the geometry, boundaries and suchlike. In the base class, this is a pipe. """

        self.obstacles = np.fromfunction(lambda xx, yy: 0, (self.nodes[0], self.nodes[1]))

        self.flow = np.fromfunction(lambda dd, xx, yy: (1-dd) * self.velocity * (1 + 1e-4*np.sin(yy/self.nodes[1]*2*np.pi)), (2, self.nodes[0], self.nodes[1]))

        #initial distributions
        self.distributions = self.equilibrium( 1.0, self.flow)
    def applyGeometry(self, velocities, density, density_equilibrium):
        #This is the left-boundary dirichlet condition
        velocities[:, 0, :] = self.flow[:, 0, :]

        #compute density from known pop
        sum_special1 = self.sum_population(self.distributions[self.special[1], 0, :]) 
        sum_special2 = 2 * self.sum_population(self.distributions[self.special[0], 0, :])
        density[0, :] = 1/(1-velocities[0,0,:]) *  sum_special1 + 2 * sum_special2
        
        # Zou/He boundary
        self.distributions[self.special[2], 0, :] = density_equilibrium[self.special[2], 0, :]

        return velocities, density
    def evolve(self): 
        """Streaming step. Returns velocity and pressure profiles. """

        self.distributions[self.special[0], -1, :] = self.distributions[self.special[0], -2, :]

        density = self.sum_population(self.distributions) 

        velocities = np.dot( self.basis_velocity.transpose(), self.distributions.transpose(1,0,2))/density
        velocities[:, self.obstacles] = 0.0

        density_equilibrium = self.equilibrium(density, velocities)

        # initial flow on left-most cells
        velocities, density = self.applyGeometry(velocities, density, density_equilibrium)

        new_distributions = self.collide(density_equilibrium)
        
        #Bounce-Back no-slip distribution
        for i in range(9):
            new_distributions[i, self.obstacles] = self.distributions[ self.noSlip[i], self.obstacles]
        
        #Stream
        self.distributions = self.stream(new_distributions)
        

        speed = np.sqrt(velocities[0]**2 + velocities[1]**2)

        return speed, density
    def stream(self, new_distributions): 
        return numba_roll(new_distributions, self.basis_velocity) 
    def collide(self, density_equilibrium):
        """ Collision step relaxation"""
        return numba_collide(self.distributions, self.relaxation_time, density_equilibrium)