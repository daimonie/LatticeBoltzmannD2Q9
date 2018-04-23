import numpy as np;
from latticeboltzmann import latticeBoltzmann;

pipe = latticeBoltzmann(1, 500, 200, 0.5)

pipe.initialise();
pipe.geometry ();