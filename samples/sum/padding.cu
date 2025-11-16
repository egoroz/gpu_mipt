#include <iostream>


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
    int rows = N / n_threads;
    
    float sum = 0;
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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    float *dA = nullptr, *dPartial = nullptr;
    cudaMalloc(&dA, paddedN * sizeof(float));
    cudaMalloc(&dPartial, n_threads * sizeof(float));
    cudaMemcpy(dA, hA, paddedN * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    reduceKernel<<<n_blocks, n_threads>>>(dA, dPartial, paddedN);
    cudaEventRecord(stop);

    cudaMemcpy(hPartial, dPartial, n_threads * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop); // Можно без нее т.к. синхронизация есть в cudaMemcpy
    float time_ms = 0;
    cudaEventElapsedTime(&time_ms, start, stop);

    float result = 0;
    for(int i = 0; i < n_threads; ++i){
        result += hPartial[i];
    }

    std::cout << "Result = " << result << "; Time = " << time_ms << std::endl;
}