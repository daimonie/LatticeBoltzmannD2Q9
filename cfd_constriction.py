{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from karman import karmanVortexSheet\n",
    "import matplotlib.pyplot as plt\n",
    "from matplotlib import cm\n",
    "from matplotlib.animation import FuncAnimation\n",
    "from numba import jit"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "re = 500\n",
    "\n",
    "numSteps = 250000\n",
    "numSnap = 100;"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "finegrain = 1\n",
    "\n",
    "width = int(finegrain * 1000)\n",
    "height = int(finegrain * 100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "Karman = karmanVortexSheet(re, width, height, height/9)\n",
    "\n",
    "Karman.initialise()\n",
    "Karman.geometry (1/10) \n",
    "\n",
    "time = 0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Iteration 0, elapsed time 3.311, speed = 0.000e+00\n",
      "Iteration 100, elapsed time 11.496, speed = 8.699e+00\n",
      "Iteration 200, elapsed time 20.519, speed = 9.747e+00\n",
      "Iteration 300, elapsed time 27.093, speed = 1.107e+01\n",
      "Iteration 400, elapsed time 34.014, speed = 1.176e+01\n",
      "Iteration 500, elapsed time 41.806, speed = 1.196e+01\n",
      "Iteration 600, elapsed time 48.355, speed = 1.241e+01\n",
      "Iteration 700, elapsed time 55.211, speed = 1.268e+01\n",
      "Iteration 800, elapsed time 61.547, speed = 1.300e+01\n",
      "Iteration 900, elapsed time 68.587, speed = 1.312e+01\n",
      "Iteration 1000, elapsed time 75.021, speed = 1.333e+01\n",
      "Iteration 1100, elapsed time 80.908, speed = 1.360e+01\n",
      "Iteration 1200, elapsed time 86.934, speed = 1.380e+01\n",
      "Iteration 1300, elapsed time 94.259, speed = 1.379e+01\n",
      "Iteration 1400, elapsed time 100.760, speed = 1.389e+01\n",
      "Iteration 1500, elapsed time 107.023, speed = 1.402e+01\n",
      "Iteration 1600, elapsed time 113.926, speed = 1.404e+01\n",
      "Iteration 1700, elapsed time 120.937, speed = 1.406e+01\n",
      "Iteration 1800, elapsed time 127.237, speed = 1.415e+01\n",
      "Iteration 1900, elapsed time 133.393, speed = 1.424e+01\n",
      "Iteration 2000, elapsed time 140.745, speed = 1.421e+01\n",
      "Iteration 2100, elapsed time 147.686, speed = 1.422e+01\n",
      "Iteration 2200, elapsed time 155.073, speed = 1.419e+01\n",
      "Iteration 2300, elapsed time 161.687, speed = 1.422e+01\n",
      "Iteration 2400, elapsed time 171.715, speed = 1.398e+01\n",
      "Iteration 2500, elapsed time 178.088, speed = 1.404e+01\n",
      "Iteration 2600, elapsed time 185.202, speed = 1.404e+01\n",
      "Iteration 2700, elapsed time 192.079, speed = 1.406e+01\n",
      "Iteration 2800, elapsed time 197.832, speed = 1.415e+01\n",
      "Iteration 2900, elapsed time 203.657, speed = 1.424e+01\n",
      "Iteration 3000, elapsed time 209.397, speed = 1.433e+01\n",
      "Iteration 3100, elapsed time 215.316, speed = 1.440e+01\n",
      "Iteration 3200, elapsed time 221.062, speed = 1.448e+01\n",
      "Iteration 3300, elapsed time 227.071, speed = 1.453e+01\n",
      "Iteration 3400, elapsed time 232.822, speed = 1.460e+01\n"
     ]
    }
   ],
   "source": [
    "while True:\n",
    "    speed, density = Karman.evolve()\n",
    "    if time%100 == 0:\n",
    "        print(\"Iteration %d, elapsed time %.3f, speed = %.3e\" % (time, Karman.report_time (), time/Karman.report_time()))\n",
    "    if time%numSnap == 0: \n",
    "        fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True,  figsize=(40, 20))\n",
    "        plt.subplot(121)\n",
    "        plt.imshow(speed.transpose(), cmap=cm.hot)\n",
    "        \n",
    "        plt.subplot(122)\n",
    "        plt.imshow(density.transpose(), cmap=cm.cool)\n",
    "        plt.xlabel(\"x\")\n",
    "        plt.ylabel(\"y\")\n",
    "        plt.title(\"Cylindrical Karman, Re = %.3f, iteration %d\" % (Karman.reynolds, time))\n",
    "        plt.savefig(\"fig/KarmanReynolds%diteration%d.png\" % (int(Karman.reynolds), int(time)))\n",
    "        plt.close()\n",
    "    time += 1"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    " <h2> Context </h2>\n",
    " <p> Some information can be found <a href=\"\">here</a>.</p>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
