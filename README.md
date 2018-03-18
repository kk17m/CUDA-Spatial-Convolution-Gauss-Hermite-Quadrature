# Fast-Spatial-Convolution-Gauss-Hermite-Quadrature
CUDA based fast numerical method to compute two dimensional spatial convolution of sufficiently smooth continuous functions using Gauss Hermite quadrature rule. 

## Description

CUDA based parallel computation 

* Inpu

* The nodes and weights are stored as list of lists using bidimensional arrays. The Golub–Welsch algorithm can be used to compute the Hermite nodes (roots of the Hermite polynomials) in the interval (-inf, inf).  

### Future Extensions

* Fast implementation of the Golub–Welsch algorithm to generate the Hermite nodes and weights for the Gauss-Hermite quadrature. 

## License & Copyright
Licensed under the [MIT License](LICENSE)
