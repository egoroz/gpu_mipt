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

__global__ void reduceKernel(const float* __restrict__ d_in, float* __restrict__ d_out, size_t N){
    extern __shared__ float sh_data[];  // blockDim * sizeof(float)  L1 Cache
    size_t tid_local = threadIdx.x;
    size_t tid_global = threadIdx.x + blockIdx.x * blockDim.x;

    float sum = 0.0f;
    if(tid_global < N) sum += d_in[tid_global];
    sh_data[tid_local] = sum;
    __syncthreads();

    for(size_t stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if(tid_local < stride){
            sh_data[tid_local] += sh_data[tid_local + stride];
        }
        __syncthreads();
    }
    if(tid_local == 0){
        d_out[blockIdx.x] = sh_data[0];
    }
}

/*
Nvidia T4 (40 SM, L1 Cache = 64 KB / SM  => 2560 KB total) google colab

Full work (data transport + calculations)
GPU res = 5e+12; Time = 9.96733 ms
CPU res = 5.0815e+12; Time = 28.9601 ms
Boost(time CPU/GPU) = 2.90551

Only calculations
GPU res = 5e+12; Time = 1.08499 ms
CPU res = 5.0815e+12; Time = 28.7362 ms
Boost(time CPU/GPU) = 26.4852
*/
int main(int argc, char** argv){
    size_t n_threads = 256;          
    size_t N = 1e7;

    float* hA = static_cast<float*>(malloc(N * sizeof(float)));

    fill_array(N, hA);
    for(int i = N; i < N; ++i) hA[i] = 0.0f;

    cudaEvent_t start_gpu, stop_gpu;
    CUDA_CHECK(cudaEventCreate(&start_gpu));
    CUDA_CHECK(cudaEventCreate(&stop_gpu));
    
    CUDA_CHECK(cudaEventRecord(start_gpu));  // start time GPU
    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, hA, N * sizeof(float), cudaMemcpyHostToDevice));
    
    size_t currentN = N;
    while(true){
        size_t grid_sz = (currentN - 1) / n_threads + 1;
        size_t shared_bytes = n_threads * sizeof(float);

        reduceKernel<<<grid_sz, n_threads, shared_bytes>>>(d_in, d_out, currentN);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        currentN = grid_sz;
        if(currentN <= 1){break;}
        
        std::swap(d_in, d_out);
    }
    
    float result = 0;    
    CUDA_CHECK(cudaEventRecord(stop_gpu));  // end time GPU
    CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost));  

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

    cudaFree(d_in);
    cudaFree(d_out);
    
    free(hA);
}