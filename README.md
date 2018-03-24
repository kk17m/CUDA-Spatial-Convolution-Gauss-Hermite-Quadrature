# Fast-Spatial-Convolution-Gauss-Hermite-Quadrature
CUDA based fast numerical method to compute two dimensional spatial convolution of sufficiently smooth continuous functions using Gauss Hermite quadrature rule. 

## Description

CUDA based parallel computation of the 2D spatial convolution. The Thrust C++ tempelate libraries are used here for abstraction and performace. The spatial convolution algorithm is implemented in the CUDA kernel file [SpatialConvolutionGHQ.cu](SpatialConvolutionGHQ.cu) and the Jupyter nootbook [Spatial_Convolution_GHQ.ipynb](Spatial_Convolution_GHQ.ipynb).

* The Python 3 Jupyter notebook [Spatial_Convolution_GHQ.ipynb](Spatial_Convolution_GHQ.ipynb) can be used on the Google colaboratory cloud service that provides a free access to the NVIDIA Tesla K80 GPU. 

  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kk17m/CUDA-Spatial-Convolution-Gauss-Hermite-Quadrature/blob/master/Spatial_Convolution_GHQ.ipynb)

* The nodes and weights are stored as list of lists using bidimensional arrays. The Golub–Welsch algorithm is used to compute the Hermite nodes (roots of the Hermite polynomials) in the interval (-inf, inf). The mathematica nootbook [Nodes-weights-Gauss-Hermite.nb](Nodes-weights-Gauss-Hermite.nb) is supplied in order to compute the Gauss-Hermite nodes and weights. 
*NOTE: Here the weighting function* exp(x^2) *is absorbed into the weights w_i.*  

### Future Extensions

* Fast implementation of the Golub–Welsch algorithm to directly generate the Hermite nodes and weights for the Gauss-Hermite quadrature. 

## License & Copyright
Licensed under the [MIT License](LICENSE)
