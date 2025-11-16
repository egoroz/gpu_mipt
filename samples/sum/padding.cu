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

__global__ void reduceKernel(float* dA, float* dPartial, size_t N){
    int tid = threadIdx.x;

    int n_threads = blockDim.x;
    size_t rows = N / n_threads;
    
    dPartial[tid] = 0;
    for(int i = 0; i < rows; ++i){
        dPartial[tid] += dA[tid + i * n_threads];
    }
}


int main(int argc, char** argv){
    size_t n_blocks = 1;
    size_t n_threads = 256;  // may vary

    size_t N = 1e6;
    size_t paddedN = ((N + n_threads - 1) / n_threads) * n_threads; 

    float* hA = static_cast<float*>(malloc(paddedN * sizeof(float)));
    float* hPartial = static_cast<float*>(malloc(n_threads * sizeof(float)));

    fill_array(N, hA);
    for(int i = N; i < paddedN; ++i) hA[i] = 0.0f;

    cudaEvent_t start_gpu, stop_gpu;
    CUDA_CHECK(cudaEventCreate(&start_gpu));
    CUDA_CHECK(cudaEventCreate(&stop_gpu));


    float *dA = nullptr, *dPartial = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, paddedN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dPartial, n_threads * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA, paddedN * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(start_gpu));
    reduceKernel<<<n_blocks, n_threads>>>(dA, dPartial, paddedN);
    CUDA_CHECK(cudaEventRecord(stop_gpu));

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(hPartial, dPartial, n_threads * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventSynchronize(stop_gpu)); // Можно без нее т.к. синхронизация есть в cudaMemcpy
    float gpu_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start_gpu, stop_gpu));

    float result = 0;
    for(int i = 0; i < n_threads; ++i){
        result += hPartial[i];
    }

    std::cout << "GPU res = " << result << "; Time = " << gpu_ms << " ms" << std::endl;


    float cpu_sum = 0.0;

    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; ++i) {
        cpu_sum += 0.1 * static_cast<double>(i);
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_ms = end_cpu - start_cpu;

    std::cout << "CPU res = " << cpu_sum << "; Time = " << cpu_ms.count() << " ms" << std::endl;
    std::cout << "Boost(GPU/CPU) = " << cpu_ms.count() / gpu_ms << std::endl;

    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    cudaFree(dA);
    cudaFree(dPartial);
    
    free(hA);
    free(hPartial);

}