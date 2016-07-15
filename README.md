# Lagrangian Relaxation Implementation

This code is an implementation of Lagrangian Relaxation to approximate the Maximal Covering problem, where we wish to find the location of P different facilities that will maximize covered demand.
Inputs include:
P: number of facilities to locate
D: maximum distance allowed from a customer to its nearest facility
among others.

Given a node demand and distance dataset, the algorithm will produce an indication at which nodes to locate facilities in order to maximize demand coverage. The optimality gap achieved is also calculated.

The graph below shows an analysis on the algorithm solution for each demand node file with a Dc value of 200. The algorithm achieves close to 95% coverage of the demand nodes at it's highest value.

![alt tag](https://github.com/ksiegler1/LagrangianRelaxation/pct_cvg.png)
