import numpy as np
from latticeboltzmann import latticeBoltzmann
import matplotlib.pyplot as plt
from matplotlib import cm

re = 10000

numSteps = 2000
numSnap = 100
pipe = latticeBoltzmann(re, 500, 200, 0.5)

pipe.initialise()
pipe.geometry ()

for time in range(numSteps):
	speed, density = pipe.evolve()
	print("Iteration %d, elapsed time %.3f" % (time, pipe.report_time ()))

	if time%numSnap == 0:
		plt.clf ()
		plt.imshow(speed.transpose(), cmap=cm.Spectral)
		plt.xlabel("x")
		plt.ylabel("y")
		plt.title("Cylindrical Pipe, Re = %.3f" % pipe.reynolds)
		plt.savefig("Re%d/pipe_%s.png" % (pipe.reynolds, time));