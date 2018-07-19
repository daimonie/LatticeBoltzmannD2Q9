################################################################################
#                                                                              #
#                        SIMPLE PYTHON LATTICE-BOLTZMANN                       #
#                                                                              #
#                            A DIPOLAR MICROSWIMMER                            #
#                                                                              #
#                                  J. DE GRAAF                                 #
#                                                                              #
#                           FOR DRSTP 2018 AT DALFSEN                          #
#                                                                              #
#                    LB CORE FROM: LBMFLOWAROUNDCYLDINER.PY                    #
#                                                                              #
################################################################################

### LOAD PYTHON LIBRARIES ######################################################

# Numerical methods from python
from numpy import *

# Linear algebra library
from numpy.linalg import *

# Import mathematical visualization (color map)
import matplotlib.pyplot as plt 
from matplotlib import cm

### DEFINITIONS OF THE FLOW PARAMETERS #########################################

maxIter = 1000                    # Run length
nSnap   = 100                     # Take a snapshot every nSnap time steps

q       = 9                       # Number of populations (D2Q9)
nx      = 100                     # Number of nodes in x direction
ny      = 100                     # Number of nodes in y direction

nulb    = 1.0                     # Kinematic viscosity
omega   = 1.0/(3.0*nulb + 0.5)    # Relaxation parameter
tau     = 1.0/omega               # Relaxation time

cdx     = nx/2                    # Center of monopole in x
cdy     = ny/2                    # Center of monopole in y
Fmono   = 1.0e-05                 # Monopole force
rblob   = 3.0                     # Extent of the force interpolation

### CONSTANTS ASSOCIATED WITH THE LATTICE ######################################

# Velocities (directions of streaming)
c = array([(x,y) for x in [0,-1,1] for y in [0,-1,1]])

# Associated weigths: 4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36
t = (1.0/36.0)*ones(q)
t[asarray([norm(ci)<1.1 for ci in c])] = 1.0/9.0
t[0] = 4.0/9.0

### FUNCTION DEFINITIONS #######################################################

# Sums the populations
sumpop = lambda fin: sum(fin,axis=0)

# Determines the equilibrium distribution
def equilibrium(rho,u):
    cu   = 3.0*dot(c,u.transpose(1,0,2))
    usqr = (3.0/2.0)*(u[0]**2 + u[1]**2)
    feq = zeros((q,nx,ny))
    for i in range(q): feq[i,:,:] = rho*t[i]*(1.0 + cu[i] + 0.5*cu[i]**2 - usqr)
    return feq

### SET UP THE INITIAL CONDITION ###############################################

# Impose quiescent fluid at t=0
vel = fromfunction(lambda d,x,y: (1-d)*0.0,(2,nx,ny))

# Determine the initial equilibrium and 'previous step' values
feq = equilibrium(1.0,vel)
fin = feq.copy()

### LATTICE BOLTZMANN INTEGRATION LOOP #########################################

# Loop over time steps
for time in range(maxIter):

    # Calculate the density from the populations
    rho = sumpop(fin)

    # Calculate the velocity from the populations
    u = dot(c.transpose(), fin.transpose((1,0,2)))/rho

    # Add the monopole force onto the fluid
    
    monof = fromfunction(lambda x,y: (x - cdx)**2.0 + (y - cdy)**2.0 < rblob**2, (nx,ny))
    monoweight = float(sum(monof))

    u[0,monof] += tau*(Fmono/monoweight)/rho[monof]

    # Ensure momentum conservation (periodic boundary conditions)

    u[0,:,:] -= tau*(Fmono/float(nx*ny))/rho[:,:]

    # Update the equilibrium populations
    feq = equilibrium(rho,u)

    # Collision step
    fout = fin - omega * (fin - feq)

    # Stream the internal populations according to the LB velocities
    for i in range(q):
        fin[i,:,:] = roll(roll(fout[i,:,:],c[i,0],axis=0),c[i,1],axis=1)

    # Visualize the magnitude of the flow field
    if ( time%nSnap == 0 ):
        plt.clf(); 
        plt.imshow(sqrt(u[0]**2 + u[1]**2).transpose(),cmap=cm.Spectral)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Stokes Flow around a Force Monopole")
        plt.savefig("monopole_vel_"+str(time/nSnap).zfill(1)+".png")

    # Export the flow field for post-processing
    if ( time%nSnap == 0 ):
        file = open("monopole_field_"+str(time/nSnap).zfill(1)+".dat","w")

        for x in range(u.shape[1]):
            for y in range(u.shape[2]):
                file.write("{} {} {} {}\n".format(x,y,u[0,x,y],u[1,x,y]))
 
        file.close()

# End program

exit(0)

################################################################################
