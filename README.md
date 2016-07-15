# Lagrangian Relaxation Implementation

This code is an implementation of Lagrangian Relaxation to approximate the Maximal Covering problem, where we wish to find the location of P different facilities that will maximize covered demand.
Inputs include:
P: number of facilities to locate
$$h_i$$: demand at node i
D: maximum distance allowed from a customer to its nearest facility

Given a node demand and distance dataset, the algorithm will produce an indication at which nodes to locate facilities in order to maximize demand coverage. The optimality gap achieved is also returne