# Fourier Tree Growing Implementation

This repository provides an implementation of Fourier Tree Growing (FTG) in Python. 
FTG is a pioneering approach to symbolic regression by considering functional analyis (FA) for optimization. 
The symbolic regression problem is reformulated from a perspective of FA and the optimization process is performed in Hilbert space. 
FTG uses a Gram matrix to retransform best solution that has been obtained back into symbolic space. 
FTG is inspired by genetic programming (GP) and evolutionary algorithms (EA). It genererates a candidate program that is represented as parse-tree by
chance with the ramped half-n-half method that is commonly used in GP. The candidate program is then optimized in Hilbert space. 
The basic design of FTG is thus based on the (1+1)-EA.
