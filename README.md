# LatticeBoltzmannD2Q9

A simple implementation of the lattice boltzmann method (d2q9) in python3. 

The Lattice Boltzmann method is a (relatively new) method for numerically
solving the Navier-Stokes equations on modern computing architectures.

#Algorithm

All simulations are on a 2D rectangular grid. Think of this as the cross-section of a pipe. 

The characteristic length is "1". Given the sound speed of sqrt(1/3), I will define the characteristic speed as sqrt(1/3)/10. Then, I can choose the reynolds number and calculate the viscosity.




# Design

Ideally, a physicist wants to do physics. The numerical solution should be as simple as possible.
Therefore, three functions will define the numerical solution:

- Compose Geometry.
	In this function, the initial grid is composed and the 'obstacles' will be defined.
	The nature of the proposed questions means that obstacles are regions of zero-density
	and no-slip boundary conditions (i.e. zero flow speed on their edge). Additionally,
	boundaries for the grid are composed here. 

- Evolution
	This is the 'streaming' step of the LB method. In this step, boundaries are applied
	and the evolution equation is used. It reads that the new densities are equal to the
	old densities minus the relaxation to equilibrium. This is hard to explain and a great
	many works on CFD will explain it.

- Export
	This is an export function. I want to define a number of these, so that switching between
	e.g. plotting realtime, saving images or saving data becomes rather simple.

# What about data?

The information on flow velocity and density defines the solution to the navier stokes equations.
Therefore, analysis of the fluid data is where the physics happens!

# Some reminders (primarily for myself)

https://www.python.org/dev/peps/pep-0020/

JIT compilation and parallelism
https://numba.pydata.org
https://numba.pydata.org/numba-doc/dev/user/parallel.html
https://numba.pydata.org/numba-doc/dev/reference/jit-compilation.html#numba.vectorize