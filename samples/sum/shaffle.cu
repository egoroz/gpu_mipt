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


__forceinline__ __device__ float warpReduceSum(float val){
    unsigned int mask = 0xffffff;
    for(size_t offset = warpSize / 2; offset > 0; offset >>= 1)

        val += __shfl_down_sync(mask, val, offset);
    return val;
}


__inline__ __device__ float blockReduceSum(float val){
    __shared__ float sum_warp[32];
    size_t lane_id = threadIdx.x & (warpSize - 1);
    size_t warp_id = threadIdx.x / warpSize;

    val = warpReduceSum(val);
    if(lane_id == 0) sum_warp[warp_id] = val;

    __syncthreads();
    val = threadIdx.x < (blockDim.x / warpSize) ? sum_warp[lane_id] : 0.0f;

    if(warp_id == 0) val = warpReduceSum(val);

    return val;

}

__global__ void kernelReduceSum(const float* __restrict__ dA, float* __restrict__ dSum, size_t N){
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0;
    for(size_t i = tid; i < N; i += blockDim.x * gridDim.x)
        sum += dA[i];
    
    sum = blockReduceSum(sum);
    if (threadIdx.x == 0) atomicAdd(&dSum[0], sum);
}

/*
Nvidia T4 (40 SM) google colab

Full work (data transport + calculations)
GPU res = 5e+12; Time = 21.0164 ms
CPU res = 5e+12; Time = 29.8656 ms
Boost(time CPU/GPU) = 1.42106

Only calculations
GPU res = 5e+12; Time = 10.5654 ms
CPU res = 5e+12; Time = 28.9812 ms
Boost(time CPU/GPU) = 2.74303
*/
int main(int argc, char** argv){
    size_t N = 1e7;

    size_t n_threads = 512;
    size_t n_blocks = (N - 1) / n_threads + 1;


    float* hA = static_cast<float*>(malloc(N * sizeof(float)));
    float* hSum = static_cast<float*>(malloc(sizeof(float)));

    fill_array(N, hA);

    cudaEvent_t start_gpu, stop_gpu;
    CUDA_CHECK(cudaEventCreate(&start_gpu));
    CUDA_CHECK(cudaEventCreate(&stop_gpu));
    
    CUDA_CHECK(cudaEventRecord(start_gpu));  // start time GPU
    float *dA = nullptr, *dSum = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dSum, sizeof(float)));
    CUDA_CHECK(cudaMemset(dSum, 0, n_blocks * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA, N * sizeof(float), cudaMemcpyHostToDevice));
    
    kernelReduceSum<<<n_blocks, n_threads>>>(dA, dSum, N);
    
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaMemcpy(hSum, dSum, sizeof(float), cudaMemcpyDeviceToHost));
    
    float result = hSum[0];
    CUDA_CHECK(cudaEventRecord(stop_gpu));  // end time GPU
    
    CUDA_CHECK(cudaEventSynchronize(stop_gpu)); // Можно без нее т.к. синхронизация есть в cudaMemcpy
    float gpu_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start_gpu, stop_gpu));

    
    std::cout << "GPU res = " << result << "; Time = " << gpu_ms << " ms" << std::endl;


    double cpu_sum = 0.0;
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
    cudaFree(dSum);
    
    free(hA);
    free(hSum);

}