import numpy as np 
from lbm import common as helper

#simple class for lbm, based on palabos notebooks
class lattice_boltzmann:
    """Class lattice_boltzmann holds the logic for a D2Q9 lbm simulation.

    Public functions:
        __init__(config): initialises object/parameters
        set_obstacle(anonymous_function): sets obstacles in the 2D pipe
        set_equilibrium(anonymous_function): sets equilibrium distribution calculation function
        set_inflow_velocity(anonymous_function): sets the inflow profile
        initialise(): initialises simulation, locks setters
        get_velocities(): returns velocity field
        get_density(): returns density 
    The Single reason to change is when the calculation itself changes.
        The handling of obstacles is through the set_obstacle method,
        while the handling of the dirichlet inflow boundary is by the
        set_inflow method, and the handling of the equilibrium distribution
        is by the set_equilibrium function.
    """
    def __init__(self, config): 
        """Initialise simulation. config is a dictionary

        Keyword arguments:
            config -- A dictionary holding numerous parameters.
                In the case of missing keys, an AssertionError is thrown.
        """

        #check the properties of the dict
        assert("reynolds" in config)
        assert("horizontal" in config)
        assert("vertical" in config)
        assert("scale" in config)
        assert("speed" in config) 

        #store these configuration parameters

        #Reynolds number and length scale used
        self.reynolds = config["reynolds"]
        self.scale = config["scale"]

        #horizontal and vertical number of 2D lattice points
        self.horizontal = config["horizontal"]
        self.vertical = config["vertical"]

        #The speed in lattice units.
        self.speed = config["speed"]

        #Set dependent configuration parameters 
        self.viscosity = self.speed * self.scale / self.reynolds
        self.relaxation = 1.0 / (3.*self.viscosity+0.5)

        #if unlocked, you can access the set_* methods.
        self.unlocked = True

        self.__lattice_init();
    def __lattice_init(self):
        """(Private) initialises the lattice, setting boundaries and basis vectors"""
        

        #Get the lattice vectors and weights. These are well known.
        self.lattice_velocities, self.weights = helper.basis_d2q9()

        self.noslip = [self.lattice_velocities.tolist().index((-self.lattice_velocities[i]).tolist()) for i in range(9)] 
        
        # Right wall; densities associated with fluid moving in from the right should be zero
        self.boundary_right = np.arange(9)[np.asarray([ci[0]<0  for ci in self.lattice_velocities])] 
        # Vertical middle. Densities associated with fluid moving horizontall should be zero.
        self.boundary_vertical = np.arange(9)[np.asarray([ci[0]==0 for ci in self.lattice_velocities])] 
        # Unknown on left wall. Densities associated with fluid moving in from the left should be zero.
        self.boundary_left = np.arange(9)[np.asarray([ci[0]>0  for ci in self.lattice_velocities])] 
    def set_obstacle(self, anonymous_obstacle):
        """sets the obstacle using the given anonymous function"""
        assert(self.unlocked)

        self.obstacle = np.fromfunction(anonymous_obstacle, (self.horizontal, self.vertical))
    def set_equilibrium(self, anonymous_equilibrium):
        """sets the _equilibrium using the given anonymous function"""
        assert(self.unlocked)

        self.equilibrium = anonymous_equilibrium
    def set_inflow_velocity(self, anonymous_velocity):
        """sets the inflow_velocity using the given anonymous function"""
        assert(self.unlocked)

        self.initial_velocity = np.fromfunction(anonymous_velocity, (2, self.horizontal, self.vertical))
    def initialise(self):
        self.unlocked = False 

        self.distributions_equilibrium = self.equilibrium(1.0, self.initial_velocity)
        self.distributions = self.distributions_equilibrium.copy()
    def propagate(self):
    
        #outflow conditions (right wall)
        self.distributions[self.boundary_right, -1, :] = self.distributions[self.boundary_right, -2, :]

        #helper function
        sumpop = lambda fin: np.sum(fin, axis=0)

        #calculate macroscopic density
        self.density = sumpop(self.distributions)

        #calculate velocity on each lattice point
        self.velocities =  np.dot(self.lattice_velocities.transpose(), self.distributions.transpose((1, 0, 2)))/self.density

        #apply dirichlet inflow boundary
        self.velocities[:, 0, :] = self.initial_velocity[:,0,:]

        #re-compute the inflow boundary densities
        self.density[0, :] = 1./(1.-self.velocities[0,0,:]) * (sumpop(self.distributions[self.boundary_vertical, 0, :])+2.*sumpop(self.distributions[self.boundary_right, 0, :]))

        current_equilibrium = self.equilibrium(self.density, self.velocities)
        
        # Left wall: Zou/He boundary condition.
        self.distributions[self.boundary_left, 0, :] = self.distributions[self.boundary_right, 0, :] + current_equilibrium[self.boundary_left, 0, :] - self.distributions[self.boundary_left, 0, :]        

        # Collision
        distributions_new = self.distributions - self.relaxation * (self.distributions - current_equilibrium)

        #apply noslip boundary. lattice velocity vectors pointing outwards are now pointing inwards (bounce back)
        for i in range(9):
            distributions_new[i, self.obstacle] = self.distributions[self.noslip[i], self.obstacle]

        #Streaming
        for i in range(9):
            self.distributions = np.roll(np.roll( distributions_new[i,:,:], self.lattice_velocities[i, 0], axis=0), c[i, 1], axis=1)


    def get_velocities(self):
        """Returns velocity field"""
        return self.velocities
    def get_densities(self):
        """Returns density """
        return self.density