import numpy as np
from karman import karmanVortexSheet
import matplotlib.pyplot as plt
from matplotlib import cm

re = 10

numSteps = 10000
numSnap = int(numSteps/100);

Karman = karmanVortexSheet(re, 500, 200, 0.5)

Karman.initialise()
Karman.geometry ()

for time in range(numSteps):
	speed, density = Karman.evolve()
	print("Iteration %d, elapsed time %.3f" % (time, Karman.report_time ()))

	if time%numSnap == 0:
		plt.clf ()
		plt.imshow(speed.transpose(), cmap=cm.Spectral)
		plt.xlabel("x")
		plt.ylabel("y")
		plt.title("Cylindrical Karman, Re = %.3f" % Karman.reynolds)
		plt.savefig("Karman%d/Karman_%s.png" % (Karman.reynolds, time));