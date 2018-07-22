import numpy as np 
from numba import jit

@jit(parallel=True)
def equilibrium(mass_density, speed, lattice_velocities, horizontal, vertical): 
    lattice_speed   = 3.0 * np.dot(lattice_velocities, speed.transpose(1,0,2))
    speed_squared = 3./2.*(speed[0]**2+speed[1]**2)
    equilibrium_densities = zeros((9, horizontal, vertical))
    for i in range(9):
    	equilibrium_densities[i,:,:] = mass_density*t[i]*(1.+lattice_speed[i]+0.5*lattice_speed[i]**2-speed_squared)
    return equilibrium_densities 

def velocity(speed, vertical):
	return lambda x,y: (-1)*speed*(1.0+1e-4*np.sin(y/vertical*2*np.pi))
