# Fourier Tree Growing Implementation

This repository provides an implementation of Fourier Tree Growing (FTG) in Python. 
FTG is a relatively new approach to symbolic regression by considering functional analyis (FA) for optimization. 
The symbolic regression problem is reformulated from a perspective of FA and the optimization process is performed in Hilbert space. 
FTG uses a Gram matrix to retransform the best solution that has been obtained back into symbolic space. 
FTG is inspired by genetic programming (GP) and evolutionary algorithms (EA). It genererates a candidate symbolic composition of elementary functions which is represented as a parse-tree with the ramped half-n-half initialization commonly used in GP. The candidate program is then optimized in Hilbert space. 
The basic design of FTG is thus based on the concept of (1+1)-EA.
