#include <stdio.h>
#include <math.h>
#define N 1024

__global__ void kernel(float *dA) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = 2.0f * 3.1415926f * (float)idx / (float)N;
        dA[idx] = sinf(sqrtf(x));
    }
}

int main() {
    float *hA, *dA;

    hA = (float*)malloc(N * sizeof(float));
    cudaMalloc(&dA, N * sizeof(float));

    kernel<<<N/512, 512>>>(dA);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("KERNEL ERROR: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceSynchronize();

    cudaMemcpy(hA, dA, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 16; i++)
        printf("a[%d] = %.8f\n", i, hA[i]);

    free(hA);
    cudaFree(dA);
}
