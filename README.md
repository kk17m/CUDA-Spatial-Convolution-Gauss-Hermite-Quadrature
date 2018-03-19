# Fast-Spatial-Convolution-Gauss-Hermite-Quadrature
CUDA based fast numerical method to compute two dimensional spatial convolution of sufficiently smooth continuous functions using Gauss Hermite quadrature rule. 

## Description

CUDA based parallel computation of the 2D spatial convolution. The Thrust C++ tempelate libraries are used here for abstraction and performace. 

* The Python 3 Jupyter notebook [Spatial_Convolution_GHQ.ipynb](Spatial_Convolution_GHQ.ipynb) can be used on the Google colaboratory platform. 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kk17m/CUDA-Spatial-Convolution-Gauss-Hermite-Quadrature/blob/master/Spatial_Convolution_GHQ.ipynb)

* The nodes and weights are stored as list of lists using bidimensional arrays. The Golub–Welsch algorithm can be used to compute the Hermite nodes (roots of the Hermite polynomials) in the interval (-inf, inf). The mathematica nootbook [Nodes-weights-Gauss-Hermite.nb](Nodes-weights-Gauss-Hermite.nb) can be used to compute the Gauss-Hermite nodes and weights. The weighting function <a href="https://www.codecogs.com/eqnedit.php?latex=\fn_cm&space;\exp(x^2)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\fn_cm&space;\exp(x^2)" title="\exp(x^2)" /></a>  

### Future Extensions

* Fast implementation of the Golub–Welsch algorithm to generate the Hermite nodes and weights for the Gauss-Hermite quadrature. 

## License & Copyright
Licensed under the [MIT License](LICENSE)
