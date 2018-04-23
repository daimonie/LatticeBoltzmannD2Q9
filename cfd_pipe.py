import numpy as np
from latticeboltzmann import latticeBoltzmann
import matplotlib.pyplot as plt
from matplotlib import cm

numSteps = 1000
numSnap = 50
pipe = latticeBoltzmann(1, 500, 200, 0.5)

pipe.initialise()
pipe.geometry ()

for time in range(numSteps):
	speed, density = pipe.evolve()
	print("Iteration %d, elapsed time %.3f" % (time, pipe.report_time ()))

	if time%numSnap == 0:
		plt.clf ()
		plt.imshow(speed, cmap=cm.Spectral)
		plt.xlabel("x")
		plt.ylabel("y")
		plt.title("Cylindrical Pipe, Re = %.3f" % pipe.reynolds)
		plt.savefig("Re%d/pipe_%s.png" % (pipe.reynolds, time));