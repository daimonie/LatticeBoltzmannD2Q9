################################################################################
#                                                                              #
#                        SIMPLE PYTHON LATTICE-BOLTZMANN                       #
#                                                                              #
#                       FOR STOKES FLOW AROUND A CYLINDER                      #
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

maxIter = 10000                   # Run length
nSnap   = 1000                    # Take a snapshot every nSnap time steps

q       = 9                       # Number of populations (D2Q9)
nx      = 720                     # Number of nodes in x direction
ny      = 180                     # Number of nodes in y direction

cx      = nx/4                    # Center of cylinder in x
cy      = ny/2                    # Center of cylinder in y
r       = ny/9                    # Cylinder radius

nulb    = 1.0                     # Kinematic viscosity
omega   = 1.0/(3.0*nulb + 0.5)    # Relaxation parameter
tau     = 1.0/omega               # Relaxation time

uLB     = 1.0e-05                 # Flow speed

### CONSTANTS ASSOCIATED WITH THE LATTICE ######################################

# Velocities (directions of streaming)
c = array([(x,y) for x in [0,-1,1] for y in [0,-1,1]])

# Associated weigths: 4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36
t = (1.0/36.0)*ones(q)
t[asarray([norm(ci)<1.1 for ci in c])] = 1.0/9.0
t[0] = 4.0/9.0

# Bounce back boundary for no-slip condition on the cylinder
noslip = [c.tolist().index((-c[i]).tolist()) for i in range(q)] 

# Which of the velocities must be special cased
i1 = arange(q)[asarray([ci[0]<0  for ci in c])] # Unknown on right edge
i2 = arange(q)[asarray([ci[0]==0 for ci in c])] # Vertical middle
i3 = arange(q)[asarray([ci[0]>0  for ci in c])] # Unknown on left edge

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

### SET UP BOUNDARY CONDITIONS #################################################

# Mark the notes associated with the cylinder
obstacle = fromfunction(lambda x,y: (x-cx)**2+(y-cy)**2<r**2, (nx,ny))

# Impose a flat flow condition for the inlet and outlet
# Actually, it is imposed everywhere in the first step
vel = fromfunction(lambda d,x,y: (1-d)*uLB,(2,nx,ny))

# Determine the initial equilibrium and 'previous step' values
feq = equilibrium(1.0,vel)
fin = feq.copy()

### LATTICE BOLTZMANN INTEGRATION LOOP #########################################

# Loop over time steps
for time in range(maxIter):

    # Right edge: the unknown populations are determined
    # from the known populations in the adjacent cell
    # resulting in an outflow boundary condition
    fin[i1,-1,:] = fin[i1,-2,:] 

    # Calculate the density from the populations
    rho = sumpop(fin)

    # Calculate the velocity from the populations
    u = dot(c.transpose(), fin.transpose((1,0,2)))/rho
    u[:,obstacle] = 0.0 

    # Impose the original sinusoidal flow condition on the left-most cells
    u[:,0,:] = vel[:,0,:]

    # Left edge: compute density from known populations
    rho[0,:] = 1./(1.-u[0,0,:]) * (sumpop(fin[i2,0,:])+2.*sumpop(fin[i1,0,:]))

    # Update the equilibrium populations
    feq = equilibrium(rho,u)

    # Left edge: adjust the populations to their imposed equilibrium values
    # this constitutes Zou/He boundary conditions
    fin[i3,0,:] = feq[i3,0,:]

    # Collision step
    fout = fin - omega * (fin - feq)

    # Impose the bounce-back no-slip condition
    for i in range(q): 
        fout[i,obstacle] = fin[noslip[i],obstacle]

    # Stream the populations according to the LB velocities
    for i in range(q):
        fin[i,:,:] = roll(roll(fout[i,:,:],c[i,0],axis=0),c[i,1],axis=1)
 
    # Visualize the magnitude of the flow field
    if ( time%nSnap == 0 ):
        plt.clf(); 
        plt.imshow(sqrt(u[0]**2 + u[1]**2).transpose(),cmap=cm.Spectral)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Stokes Flow around Cylinder")
        plt.savefig("stokes_cyl_vel_"+str(time/nSnap).zfill(1)+".png")

# End program

exit(0)

################################################################################
