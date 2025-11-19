#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// ---------------------------------------------------------
// Функция-ядро
// ---------------------------------------------------------
__global__ void function(float *dA, float *dB, float *dC, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float ab = dA[i] * dB[i];
        float sum = 0.0f;
        for (int j = 0; j < 100; j++) {
            sum = sum + sinf(j + ab);
        }
        dC[i] = sum;
    }
}

int main() {
    // ---------------------------------------------------------
    // Инициализация
    // ---------------------------------------------------------
    float *hA, *hB, *hC, *hC_CPU; 
    float *dA, *dB, *dC; 

    int nStream = 4; 
    int total_N = 512 * 50000;
    int size = total_N / nStream; 
    int N_thread = 512;             
    int N_blocks = size / N_thread; 

    unsigned int mem_size = sizeof(float) * size;
    unsigned int total_mem_size = sizeof(float) * total_N;

    // Выделение памяти
    cudaMallocHost((void**)&hA, total_mem_size);
    cudaMallocHost((void**)&hB, total_mem_size);
    cudaMallocHost((void**)&hC, total_mem_size);
    hC_CPU = (float*)malloc(total_mem_size);

    cudaMalloc((void**)&dA, total_mem_size);
    cudaMalloc((void**)&dB, total_mem_size);
    cudaMalloc((void**)&dC, total_mem_size);

    // Заполнение данными
    printf("Initializing data...\n");
    for (int i = 0; i < total_N; i++) {
        hA[i] = sinf(i);
        hB[i] = cosf(2.0f * i - 5.0f);
        hC[i] = 0.0f;
    }

    // Создание Streams и Events
    cudaStream_t stream[4];
    for (int i = 0; i < nStream; ++i) cudaStreamCreate(&stream[i]);

    cudaEvent_t startTotal, stopTotal, startKernel, stopKernel;
    cudaEventCreate(&startTotal); cudaEventCreate(&stopTotal);
    cudaEventCreate(&startKernel); cudaEventCreate(&stopKernel);

    float timeTotal1, timeKernel1;
    float timeTotal2, timeKernel2;

    // =========================================================
    // ВАРИАНТ 1: 3 ЦИКЛА (Breadth-First) - Как на слайдах
    // =========================================================
    printf("Running GPU Variant 1 (3 separate loops)...\n");
    
    cudaEventRecord(startTotal, 0);

    // 1. Все копирования H2D
    for (int i = 0; i < nStream; ++i) {
        cudaMemcpyAsync(dA + i*size, hA + i*size, mem_size, cudaMemcpyHostToDevice, stream[i]);
    }

    // 2. Все ядра
    for (int i = 0; i < nStream; ++i) {
        if (i == 0) cudaEventRecord(startKernel, stream[i]); // Старт замера ядер
        
        function<<<N_blocks, N_thread, 0, stream[i]>>>(dA + i*size, dB + i*size, dC + i*size, size);
        
        if (i == nStream - 1) cudaEventRecord(stopKernel, stream[i]); // Стоп замера ядер
    }

    // 3. Все копирования D2H
    for (int i = 0; i < nStream; ++i) {
        cudaMemcpyAsync(hC + i*size, dC + i*size, mem_size, cudaMemcpyDeviceToHost, stream[i]);
    }

    cudaEventRecord(stopTotal, 0);
    cudaDeviceSynchronize();
    
    cudaEventElapsedTime(&timeTotal1, startTotal, stopTotal);
    cudaEventElapsedTime(&timeKernel1, startKernel, stopKernel);

    // Очистка результата на хосте перед вторым тестом
    for(int i=0; i<total_N; i++) hC[i] = 0.0f;

    // =========================================================
    // ВАРИАНТ 2: 1 ЦИКЛ (Depth-First)
    // =========================================================
    printf("Running GPU Variant 2 (1 single loop)...\n");

    cudaEventRecord(startTotal, 0);

    for (int i = 0; i < nStream; ++i) {
        int offset = i * size;

        // 1. Копирование H2D
        cudaMemcpyAsync(dA + offset, hA + offset, mem_size, cudaMemcpyHostToDevice, stream[i]);

        // 2. Ядро
        if (i == 0) cudaEventRecord(startKernel, stream[i]); // Старт замера ядер
        function<<<N_blocks, N_thread, 0, stream[i]>>>(dA + offset, dB + offset, dC + offset, size);
        if (i == nStream - 1) cudaEventRecord(stopKernel, stream[i]); // Стоп замера ядер

        // 3. Копирование D2H
        cudaMemcpyAsync(hC + offset, dC + offset, mem_size, cudaMemcpyDeviceToHost, stream[i]);
    }

    cudaEventRecord(stopTotal, 0);
    cudaDeviceSynchronize();

    cudaEventElapsedTime(&timeTotal2, startTotal, stopTotal);
    cudaEventElapsedTime(&timeKernel2, startKernel, stopKernel);

    // =========================================================
    // CPU ВАРИАНТ
    // =========================================================
    printf("Running CPU calculation...\n");
    clock_t cpu_start = clock();

    for (int k = 0; k < total_N; k++) {
        float ab = hA[k] * hB[k];
        float sum = 0.0f;
        for (int j = 0; j < 100; j++) sum += sinf(j + ab);
        hC_CPU[k] = sum;
    }

    clock_t cpu_end = clock();
    float cpu_time_ms = 1000.0f * (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC;

    // =========================================================
    // ВЫВОД РЕЗУЛЬТАТОВ
    // =========================================================
    printf("\n================================================\n");
    printf("РЕЗУЛЬТАТЫ ТЕСТОВ\n");
    printf("================================================\n");
    
    printf("CPU Time: \t\t%.0f ms\n", cpu_time_ms);
    printf("------------------------------------------------\n");
    
    // Вариант 1 (3 цикла)
    printf("[GPU 3 Loops] Kernel: \t%.0f ms \t(Total: %.0f ms)\n", timeKernel1, timeTotal1);
    printf("[GPU 3 Loops] Rate: \t%.1f x\n", cpu_time_ms / timeTotal1);
    
    printf("------------------------------------------------\n");
    
    // Вариант 2 (1 цикл)
    printf("[GPU 1 Loop]  Kernel: \t%.0f ms \t(Total: %.0f ms)\n", timeKernel2, timeTotal2);
    printf("[GPU 1 Loop]  Rate: \t%.1f x\n", cpu_time_ms / timeTotal2);
    
    printf("------------------------------------------------\n");
    printf("CUDA-Streams: %d\n", nStream);
    printf("================================================\n");

    // Очистка
    for (int i = 0; i < nStream; ++i) cudaStreamDestroy(stream[i]);
    cudaEventDestroy(startTotal); cudaEventDestroy(stopTotal);
    cudaEventDestroy(startKernel); cudaEventDestroy(stopKernel);
    cudaFreeHost(hA); cudaFreeHost(hB); cudaFreeHost(hC); free(hC_CPU);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);

    return 0;
}