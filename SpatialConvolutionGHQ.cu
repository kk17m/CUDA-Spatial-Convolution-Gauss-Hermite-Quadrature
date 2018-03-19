// *******************************************************
//   Spatial convolution using Gauss-Hermite quadrature
//
// *******************************************************

#include <iostream>
#include <math.h>
#include <fstream>
#include <vector>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

// CUDA thrust library
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

        //
        // Definig the CLOCK for performance testing.
        //
        long long wall_clock_time()
{
#ifdef __linux__
    struct timespec tp;
    clock_gettime(CLOCK_REALTIME, &tp);
    return (long long)(tp.tv_nsec + (long long)tp.tv_sec * 1000000000ll);
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long long)(tv.tv_usec * 1000 + (long long)tv.tv_sec * 1000000000ll);
#endif
}

//
// The parameters to compute the discrete points (Xn, Yn) are defined here.
// The axis limits along the x-axis are given by AXIS_MIN_X and AXIS_MAX_X, the
// axis limits along the y-axis are given by AXIS_MIN_Y and AXIS_MAX_Y.
//
// NOTE: These axis limits are not the limits of integration. The limits of
// integration are (-inf,inf).
//
// The number of discrete points Xn and Yn are given by NUM_PTS_X and NUM_PTS_Y.
// These points can have different sizes and should be a multiple of the
// BLOCK_SIZE in the respective dimension.
//
#define AXIS_MIN_X   -1
#define AXIS_MAX_X    1
#define AXIS_MIN_Y   -1
#define AXIS_MAX_Y    1
#define NUM_PTS_X 256
#define NUM_PTS_Y 256

//
// The CUDA parameters are defined here.
// The BLOCK_SIZE parameter for the CUDA x-dimension can be different than the
// CUDA y-dimension.
//
// The Z_BLOCK_SIZE should be a factor of sizeof(Gy)/sizeof(Gy[0]).
//
#define BLOCK_SIZE 16
#define Z_BLOCK_SIZE 4

//
// Define the Gauss-Hermite nodes n_i and weights w_i*exp((n_i)^2) for
// the two integrals. The size of Gy and Gx can be different depending on the
// required precision of the quadrature approximation.
//
__constant__ float Gy[36][2] = {{-7.626325754003896,0.8072646660353702},{-6.925598990259945,0.626482063538593},{-6.342243330994417,0.5482379054346118},{-5.818863279505579,0.5018952690514574},{-5.3335601071130645,0.4705108735743688},{-4.875039972467083,0.4476297837537447},{-4.436506970192858,0.4301720223313478},{-4.013456567749471,0.41645347099886904},{-3.6026938571484726,0.4054649988533432},{-3.201833945788157,0.3965612262672993},{-2.8090222351311054,0.38930924155705054},{-2.422766042053559,0.3834083398416976},{-2.0418271835544166,0.3786444980895176},{-1.6651500018434104,0.3748631855184457},{-1.2918109588209203,0.3719524810189278},{-0.9209818015707496,0.36983231208820944},{-0.5519014332904186,0.36844752436798417},{-0.18385336710581246,0.3677634858284455},{0.18385336710581512,0.36776348582843993},{0.5519014332904222,0.3684475243679883},{0.9209818015707576,0.3698323120882103},{1.2918109588209283,0.3719524810189504},{1.6651500018434149,0.3748631855184701},{2.0418271835544193,0.3786444980895354},{2.4227660420535626,0.38340833984170997},{2.8090222351311027,0.38930924155705887},{3.2018339457881595,0.3965612262673096},{3.6026938571484743,0.40546499885337384},{4.013456567749469,0.4164534709988875},{4.436506970192857,0.4301720223313582},{4.875039972467084,0.4476297837537804},{5.333560107113062,0.4705108735743989},{5.8188632795055755,0.5018952690514803},{6.342243330994413,0.5482379054346204},{6.925598990259943,0.6264820635385975},{7.6263257540038945,0.8072646660353738}};
__constant__ float Gx[36][2] = {{-7.626325754003896,0.8072646660353702},{-6.925598990259945,0.626482063538593},{-6.342243330994417,0.5482379054346118},{-5.818863279505579,0.5018952690514574},{-5.3335601071130645,0.4705108735743688},{-4.875039972467083,0.4476297837537447},{-4.436506970192858,0.4301720223313478},{-4.013456567749471,0.41645347099886904},{-3.6026938571484726,0.4054649988533432},{-3.201833945788157,0.3965612262672993},{-2.8090222351311054,0.38930924155705054},{-2.422766042053559,0.3834083398416976},{-2.0418271835544166,0.3786444980895176},{-1.6651500018434104,0.3748631855184457},{-1.2918109588209203,0.3719524810189278},{-0.9209818015707496,0.36983231208820944},{-0.5519014332904186,0.36844752436798417},{-0.18385336710581246,0.3677634858284455},{0.18385336710581512,0.36776348582843993},{0.5519014332904222,0.3684475243679883},{0.9209818015707576,0.3698323120882103},{1.2918109588209283,0.3719524810189504},{1.6651500018434149,0.3748631855184701},{2.0418271835544193,0.3786444980895354},{2.4227660420535626,0.38340833984170997},{2.8090222351311027,0.38930924155705887},{3.2018339457881595,0.3965612262673096},{3.6026938571484743,0.40546499885337384},{4.013456567749469,0.4164534709988875},{4.436506970192857,0.4301720223313582},{4.875039972467084,0.4476297837537804},{5.333560107113062,0.4705108735743989},{5.8188632795055755,0.5018952690514803},{6.342243330994413,0.5482379054346204},{6.925598990259943,0.6264820635385975},{7.6263257540038945,0.8072646660353738}};

//
// Declare the global vectors Xn, Yn, Cn here.
//
thrust::host_vector<float> Xn;
thrust::host_vector<float> Yn;
thrust::host_vector<float> Cn(NUM_PTS_X * NUM_PTS_Y);

//
// Define the function f(x,y) here.
//
__device__ float Fun(float x, float y)
{
    return exp(-(pow(x,2) + pow(y,2)));
}

//
// Define the convolution kernel g(x,y) here.
//
__device__ float Conv_Kernel(float x, float y)
{
    return exp(-(pow(x,2) + pow(y,2))/0.2);
}

//
// The inner quadrature sum, with weights wx and nodes nx, is computed here.
//
__device__ float Sum(float* ptrXn, float* ptrYn, float *ny, int *idx, int *idy)
{
    float nx, wx, Q1 = 0.0f;;

    int Nx = sizeof(Gx)/sizeof(Gx[0]);

    for (int k=0; k<Nx; k++)
    {
        nx = Gx[k][0];
        wx = Gx[k][1];
        Q1 +=  wx * Fun(nx, *ny) * Conv_Kernel(nx - ptrXn[*idx], *ny - ptrYn[*idy]) ;
    }

    return Q1;
}

//
// The CUDA kernel is defined here and the outer quadrature sum, with weights
// wy and nodes ny, is computed here.
//
__global__ void CUDA_kernel(float* ptrXn, float* ptrYn, float* ptrCn){

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int idy = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int idz = blockIdx.z * Z_BLOCK_SIZE + threadIdx.z;

    float ny, wy;
    int stride_z = blockDim.z * gridDim.z;
    int Ny = sizeof(Gy)/sizeof(Gy[0]);

    while (idz < Ny ) {
        ny = Gy[idz][0];
        wy = Gy[idz][1];
        atomicAdd( &( ptrCn[idy * NUM_PTS_X + idx]), wy * Sum(ptrXn, ptrYn, &ny, &idx, &idy));
        idz += stride_z;
    }

}

int Kernelcall(){

    thrust::device_vector<float> d_Xn = Xn;
    thrust::device_vector<float> d_Yn = Yn;
    thrust::device_vector<float> d_Cn = Cn;

    float * ptrXn = thrust::raw_pointer_cast(&d_Xn[0]);
    float * ptrYn = thrust::raw_pointer_cast(&d_Yn[0]);
    float * ptrCn = thrust::raw_pointer_cast(&d_Cn[0]);

    int Ny = sizeof(Gy)/sizeof(Gy[0]);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, Z_BLOCK_SIZE);
    dim3 dimGrid((Xn.size() + dimBlock.x - 1) / dimBlock.x, (Yn.size() + dimBlock.y - 1) / dimBlock.y, (Ny + dimBlock.z - 1) / dimBlock.z);

    CUDA_kernel<<<dimGrid, dimBlock>>>(ptrXn, ptrYn, ptrCn);
    thrust::copy(d_Cn.begin(), d_Cn.end(), Cn.begin());

    cudaError_t rc;
    rc = cudaGetLastError();
    if (rc != cudaSuccess)
        printf("Last CUDA error %s\n", cudaGetErrorString(rc));

    //
    // Save result to a file
    //
    char buffer[32]; // The filename buffer.
    snprintf(buffer, sizeof(char) * 32, "FILE%i.txt", 0);
    std::ofstream out(buffer, std::ios_base::app);
    out.setf(std::ios::scientific);
    if( !out )
    {
        std::cout << "Couldn't open file."  << std::endl;
        return 1;
    }

    for (int i = 0; i < NUM_PTS_Y; i++) {
        for (int j = 0; j < NUM_PTS_X; j++) {
            out << Cn[i * NUM_PTS_X + j] <<',';
        }
        out <<'\n';
    }

    out.close();

    return 0;
}


//
// The main() function.
//
int main(int argc, char *argv[]){

    long long before, after;
    before = wall_clock_time();                                                                     // TIME START

    float xl = AXIS_MIN_X, xr = AXIS_MAX_X, yl = AXIS_MIN_Y, yr = AXIS_MAX_Y;
    int xpix = NUM_PTS_X, ypix = NUM_PTS_Y;

    thrust::host_vector<float> Del;
    Del.push_back((xr - xl) / xpix);
    Del.push_back((yr - yl) / ypix);

    for(int i=0; i < xpix; i++){
        Xn.push_back(xl + Del[0] * (i + 0.5));
    }

    for(int i=0; i < ypix; i++){
        Yn.push_back(yl + Del[1] * (i + 0.5));
    }

    Kernelcall();

    after = wall_clock_time();                                                                      // TIME END
    fprintf(stderr, "Process took %3.5f seconds ", ((float)(after - before))/1000000000);

    return 0;
}
