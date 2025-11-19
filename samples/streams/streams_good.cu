#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// ---------------------------------------------------------
// Часть 1. Функция-ядро (Kernel)
// ---------------------------------------------------------
__global__ void function(float *dA, float *dB, float *dC, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Проверка границ
    if (i < size) {
        float ab = dA[i] * dB[i];
        float sum = 0.0f;
        // Нагрузочный цикл
        for (int j = 0; j < 100; j++) {
            sum = sum + sinf(j + ab);
        }
        dC[i] = sum;
    }
}

int main() {
    // ---------------------------------------------------------
    // Часть 2. Инициализация и выделение памяти
    // ---------------------------------------------------------
    float *hA, *hB, *hC; // Host Pinned Memory
    float *hC_CPU;       // Host Standard Memory (для CPU теста)
    float *dA, *dB, *dC; // Device Memory

    int nStream = 4; 
    int total_N = 512 * 500000; // 25,600,000 элементов
    int size = total_N / nStream; 
    
    int N_thread = 512;             
    int N_blocks = size / N_thread; 

    unsigned int mem_size = sizeof(float) * size;
    unsigned int total_mem_size = sizeof(float) * total_N;

    // Выделение Pinned Memory (Критично для асинхронности)
    cudaMallocHost((void**)&hA, total_mem_size);
    cudaMallocHost((void**)&hB, total_mem_size);
    cudaMallocHost((void**)&hC, total_mem_size);
    
    // Память для CPU теста
    hC_CPU = (float*)malloc(total_mem_size);

    // Память на GPU
    cudaMalloc((void**)&dA, total_mem_size);
    cudaMalloc((void**)&dB, total_mem_size);
    cudaMalloc((void**)&dC, total_mem_size);

    // Заполнение данных
    printf("Initializing data (%d elements)...\n", total_N);
    for (int i = 0; i < total_N; i++) {
        hA[i] = sinf(i);
        hB[i] = cosf(2.0f * i - 5.0f);
        hC[i] = 0.0f;
        hC_CPU[i] = 0.0f;
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
    // ВАРИАНТ 1: 3 ЦИКЛА (Breadth-First - Как на слайдах)
    // =========================================================
    printf("Running GPU Variant 1 (3 loops structure)...\n");
    
    // Старт общего таймера
    cudaEventRecord(startTotal, 0);

    // 1. Цикл копирования Host -> Device
    for (int i = 0; i < nStream; ++i) {
        cudaMemcpyAsync(dA + i*size, hA + i*size, mem_size, cudaMemcpyHostToDevice, stream[i]);
    }

    // 2. Цикл запуска ядер
    for (int i = 0; i < nStream; ++i) {
        // Засекаем начало работы ядер (в первом потоке)
        if (i == 0) cudaEventRecord(startKernel, stream[i]);
        
        function<<<N_blocks, N_thread, 0, stream[i]>>>(dA + i*size, dB + i*size, dC + i*size, size);
        
        // Засекаем конец работы ядер (в последнем потоке)
        if (i == nStream - 1) cudaEventRecord(stopKernel, stream[i]);
    }

    // 3. Цикл копирования Device -> Host
    for (int i = 0; i < nStream; ++i) {
        cudaMemcpyAsync(hC + i*size, dC + i*size, mem_size, cudaMemcpyDeviceToHost, stream[i]);
    }

    // Стоп общего таймера
    cudaEventRecord(stopTotal, 0);
    cudaDeviceSynchronize();
    
    cudaEventElapsedTime(&timeTotal1, startTotal, stopTotal);
    cudaEventElapsedTime(&timeKernel1, startKernel, stopKernel);

    // Очистка результата перед вторым тестом
    for(int i=0; i<total_N; i++) hC[i] = 0.0f;

    // =========================================================
    // ВАРИАНТ 2: 1 ЦИКЛ (Depth-First - Вся цепочка в одном цикле)
    // =========================================================
    printf("Running GPU Variant 2 (1 single loop)...\n");

    cudaEventRecord(startTotal, 0);

    for (int i = 0; i < nStream; ++i) {
        int offset = i * size;

        // H2D
        cudaMemcpyAsync(dA + offset, hA + offset, mem_size, cudaMemcpyHostToDevice, stream[i]);

        // Kernel
        if (i == 0) cudaEventRecord(startKernel, stream[i]);
        function<<<N_blocks, N_thread, 0, stream[i]>>>(dA + offset, dB + offset, dC + offset, size);
        if (i == nStream - 1) cudaEventRecord(stopKernel, stream[i]);

        // D2H
        cudaMemcpyAsync(hC + offset, dC + offset, mem_size, cudaMemcpyDeviceToHost, stream[i]);
    }

    cudaEventRecord(stopTotal, 0);
    cudaDeviceSynchronize();

    cudaEventElapsedTime(&timeTotal2, startTotal, stopTotal);
    cudaEventElapsedTime(&timeKernel2, startKernel, stopKernel);

    // =========================================================
    // CPU ВАРИАНТ (1 поток)
    // =========================================================
    printf("Running CPU calculation (single thread)...\n");
    
    clock_t cpu_start = clock();

    for (int k = 0; k < total_N; k++) {
        float ab = hA[k] * hB[k];
        float sum = 0.0f;
        for (int j = 0; j < 100; j++) {
            sum = sum + sinf(j + ab);
        }
        hC_CPU[k] = sum;
    }

    clock_t cpu_end = clock();
    // Перевод тиков в миллисекунды
    float cpu_time_ms = 1000.0f * (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC;

    // =========================================================
    // ВЫВОД РЕЗУЛЬТАТОВ
    // =========================================================
    printf("\n================================================\n");
    printf("РЕЗУЛЬТАТ\n");
    printf("================================================\n");
    
    // --- CPU ---
    printf("CPU calculation time: \t%.0f ms\n", cpu_time_ms);
    printf("------------------------------------------------\n");

    // --- GPU Вариант 1 ---
    float rateKernel1 = cpu_time_ms / timeKernel1;
    float rateTotal1 = cpu_time_ms / timeTotal1;

    printf("GPU (3 Loops) time: \t%.0f ms (%.0f)\n", timeKernel1, timeTotal1);
    // Вывод ускорения: Без скобок (по ядру) и в скобках (по полному времени)
    printf("Rate (3 Loops): \t%.0f x (%.0f)\n", rateKernel1, rateTotal1);
    
    printf("------------------------------------------------\n");

    // --- GPU Вариант 2 ---
    float rateKernel2 = cpu_time_ms / timeKernel2;
    float rateTotal2 = cpu_time_ms / timeTotal2;

    printf("GPU (1 Loop) time: \t%.0f ms (%.0f)\n", timeKernel2, timeTotal2);
    // Вывод ускорения: Без скобок (по ядру) и в скобках (по полному времени)
    printf("Rate (1 Loop): \t\t%.0f x (%.0f)\n", rateKernel2, rateTotal2);
    
    printf("------------------------------------------------\n");
    printf("CUDA-Streams: %d\n", nStream);
    printf("================================================\n");

    // Очистка ресурсов
    for (int i = 0; i < nStream; ++i) cudaStreamDestroy(stream[i]);
    cudaEventDestroy(startTotal); cudaEventDestroy(stopTotal);
    cudaEventDestroy(startKernel); cudaEventDestroy(stopKernel);
    
    cudaFreeHost(hA); cudaFreeHost(hB); cudaFreeHost(hC); free(hC_CPU);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);

    return 0;
}