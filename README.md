This repository contains the simulation code, benchmark data and figures used in our paper;

## Fast time evolution of matrix product states using the QR decomposition
Jakob Unfried, Johannes Hauschild and Frank Pollmann

[Phys. Rev. B 107, 155133 – Published 21 April 2023](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.107.155133)

We propose and benchmark a modified time evolution block decimation (TEBD) algorithm that uses a 
truncation scheme based on the QR decomposition instead of the singular value decomposition (SVD).
The modification reduces the scaling with the dimension of the physical Hilbert space d from $d^3$ 
down to $d^2$. Moreover, the QR decomposition has a lower computational complexity than the SVD and 
allows for highly efficient implementations on GPU hardware. In a benchmark simulation of a global
quench in a quantum clock model, we observe a speedup of up to three orders of magnitude comparing
QR and SVD based updates on an A100 GPU.

## Tenpy implementation

There is also an implementation of the QR-based algorithm in [tenpy](https://github.com/tenpy/tenpy), 
which supports symmetries, but can only run on CPU.

As of april 2023, it is included in the main branch, but not yet in an official release, so you need 
to build tenpy from source if you want to use it.

GPU support for tenpy is on the longterm roadmap..

