#include <iostream>
#include <chrono>
#include <cuda_runtime.h>


#define CUDA_CHECK(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true){
    if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s From file: %s In line: %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}

void fill_array(size_t N, float* arr){
    for(int i = 0; i < N; ++i){
        arr[i] = 0.1f * static_cast<float>(i);
    }
}

__global__ void reduceKernel(const float* __restrict__ dA, float* __restrict__ dPartial, size_t N){
    size_t shift = blockDim.x * blockIdx.x;
    int tid = threadIdx.x + shift;
    size_t row_sz = blockDim.x * gridDim.x;
    size_t rows = N / row_sz;
    
    for(int i = 0; i < rows; ++i){
        dPartial[tid] += dA[tid + i * row_sz];
    }
}

/*
Nvidia T4 (40 SM) google colab

GPU res = 4.99965e+12; Time = 9.66675 ms
CPU res = 5.0815e+12; Time = 29.1528 ms
Boost(time CPU/GPU) = 3.01578
*/
int main(int argc, char** argv){
    size_t k_SM = 40;
    size_t n_blocks = 16 * k_SM;     // may vary. 16 blocks in the SM
    size_t n_threads = 128;          // may vary
    size_t c = n_threads * n_blocks; // num columns

    size_t N = 1e7;
    size_t paddedN = ((N + c - 1) / c) * c; 

    float* hA = static_cast<float*>(malloc(paddedN * sizeof(float)));
    float* hPartial = static_cast<float*>(malloc(n_threads * n_blocks * sizeof(float)));

    fill_array(N, hA);
    for(int i = N; i < paddedN; ++i) hA[i] = 0.0f;

    cudaEvent_t start_gpu, stop_gpu;
    CUDA_CHECK(cudaEventCreate(&start_gpu));
    CUDA_CHECK(cudaEventCreate(&stop_gpu));
    
    float *dA = nullptr, *dPartial = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, paddedN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dPartial, n_threads * n_blocks * sizeof(float)));
    CUDA_CHECK(cudaMemset(dPartial, 0, n_threads * n_blocks * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA, paddedN * sizeof(float), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaEventRecord(start_gpu));  // start time GPU
    reduceKernel<<<n_blocks, n_threads>>>(dA, dPartial, paddedN);
    CUDA_CHECK(cudaEventRecord(stop_gpu));  // end time GPU
    
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaMemcpy(hPartial, dPartial, n_threads * n_blocks * sizeof(float), cudaMemcpyDeviceToHost));  
    
    float result = 0;
    for(int i = 0; i < c; ++i){
        result += hPartial[i];
    }

    CUDA_CHECK(cudaEventSynchronize(stop_gpu)); // Можно без нее т.к. синхронизация есть в cudaMemcpy
    float gpu_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start_gpu, stop_gpu));

    std::cout << "GPU res = " << result << "; Time = " << gpu_ms << " ms" << std::endl;


    float cpu_sum = 0.0;

    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; ++i) {
        cpu_sum += hA[i];
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_ms = end_cpu - start_cpu;

    std::cout << "CPU res = " << cpu_sum << "; Time = " << cpu_ms.count() << " ms" << std::endl;
    std::cout << "Boost(time CPU/GPU) = " << cpu_ms.count() / gpu_ms << std::endl;

    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    cudaFree(dA);
    cudaFree(dPartial);
    
    free(hA);
    free(hPartial);

}