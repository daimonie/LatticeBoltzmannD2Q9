################################################################################
#                                                                              #
#                        SIMPLE PYTHON LATTICE-BOLTZMANN                       #
#                                                                              #
#                    FOR HAGEN-POISSEUILLE FLOW IN A CHANNEL                   #
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

maxIter = 2000                    # Run length
nSnap   =  200                    # Take a snapshot every nSnap time steps

q       = 9                       # Number of populations (D2Q9)
nx      = 10                      # Number of nodes in x direction
ny      = 50                      # Number of nodes in y direction
ly      = ny - 1.0                # Width of the channel

nulb    = 1.0                     # Kinematic viscosity
omega   = 1.0/(3.0*nulb + 0.5)    # Relaxation parameter
tau     = 1.0/omega               # Relaxation time

F       = 1.0e-05                 # Driving force on the fluid
uLB     = 0.5*F*ly*ly/nulb        # Velocity in lattice units

### CONSTANTS ASSOCIATED WITH THE LATTICE ######################################

# Velocities (directions of streaming)
c = array([(x,y) for x in [0,-1,1] for y in [0,-1,1]])

# Associated weigths: 4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36
t = (1.0/36.0)*ones(q)
t[asarray([norm(ci)<1.1 for ci in c])] = 1.0/9.0
t[0] = 4.0/9.0

# Bounce back boundary for no-slip condition on the channel walls
noslip = [c.tolist().index((-c[i]).tolist()) for i in range(q)]

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

# Mark the notes associated with the walls and the fluid
walls = fromfunction(lambda x,y: (y < 1) != (y > ny-2) , (nx,ny))

# The initial velocity zero everywhere
vel = fromfunction(lambda d,x,y: (1-d)*0.0,(2,nx,ny))
vel[:,walls] = 0.0 

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

    # Add a homogeneous force onto the fluid nodes 
    # This force only has an x component, and is zero in the wall
    u[0,:,:] += tau*F/rho[:,:]
    u[0,walls] = 0.0 

    # Update the equilibrium populations
    feq = equilibrium(rho,u)

    # Collision step
    fout = fin - omega * (fin - feq)

    # Impose the bounce-back no-slip condition
    for i in range(q): 
        fout[i,walls] = fin[noslip[i],walls]

    # Stream the internal populations according to the LB velocities
    for i in range(q):
        fin[i,:,:] = roll(roll(fout[i,:,:],c[i,0],axis=0),c[i,1],axis=1)

    # Visualize the shape of the flow field
    if ( time%nSnap == 0 ):
        plt.clf(); 

        del_ly   = ly/50.
        xax_calc = arange(-ly/2.0,(ly+del_ly)/2.0,del_ly)
        yax_calc = zeros(len(xax_calc))

        for i in range(len(xax_calc)):
            t1 = (xax_calc[i] + ly/2.0)/ly
            t2 = (xax_calc[i] - ly/2.0)/ly
            yax_calc[i] = -uLB*t1*t2

        xax_comp = arange(-ny/2+0.5,ny/2,1)
        yax_comp = u[0,int(nx/2),:]

        plt.plot(xax_comp,yax_comp, color='blue', linestyle='--', label='LB')
        plt.plot(xax_calc,yax_calc, color='red', label='theory')
        plt.xlabel("y")
        plt.ylabel("v")
        plt.legend()

        plt.savefig("hagen_poisseuille_graph_"+str(time/nSnap).zfill(2)+".png")

# End program

exit(0)

################################################################################
