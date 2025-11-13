#include <stdio.h> 
#define N (1024)  

__global__ void kernel ( float * dA )  { 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   
    float x = 2.0f * 3.1415926f * (float) idx / (float) N;    
    dA [idx] = sinf (sqrtf ( x ) ); 
} 
 

int main ( int argc, char * argv [] )  {
    float *hA, *dA;   
    hA = ( float* ) malloc (N * sizeof ( float ) );   
    cudaMalloc ( (void**)&dA, N * sizeof ( float ) );  
    kernel <<< N/512, 512 >>> ( dA );   
    cudaMemcpy ( hA, dA, N * sizeof ( float ), cudaMemcpyDeviceToHost );  
    for ( int idx = 0; idx < N; idx++ ) printf ( "a[%d] = %.8f\n", idx, hA[idx] );   
    free ( hA ); 
    cudaFree ( dA );   
    return 0; 
} 
