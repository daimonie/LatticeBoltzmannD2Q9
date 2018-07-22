import numpy as np 
from numba import jit

@jit(parallel=True)
def equilibrium(mass_density, speed, lattice_velocities, horizontal, vertical): 
    
    basis_vectors, basis_weights = basis_d2q9()
    

    lattice_speed   = 3.0 * np.dot(lattice_velocities, speed.transpose(1,0,2))
    speed_squared = 3./2.*(speed[0]**2+speed[1]**2)
    equilibrium_densities = np.zeros((9, horizontal, vertical))
    for i in range(9):
        equilibrium_densities[i,:,:] = mass_density*basis_weights[i]*(1.+lattice_speed[i]+0.5*lattice_speed[i]**2-speed_squared)
    return equilibrium_densities 

def velocity(speed, vertical):
    return lambda d, x,y: (1-d)*speed*(1.0+1e-4*np.sin(y/vertical*2*np.pi))
def basis_d2q9():

    # The D2Q9 lattice vector basis.
    basis_vectors = np.array([(x,y) for x in [0,-1,1] for y in [0,-1,1]]) 
                                   
    # Weights of the D2Q9 basis.

    basis_weights = 1./36. * np.ones(9)    
    basis_weights[np.asarray([np.linalg.norm(ci)<1.1 for ci in basis_vectors])] = 1./9.
    basis_weights[0] = 4./9.

    return basis_vectors, basis_weights