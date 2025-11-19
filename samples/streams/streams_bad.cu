#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h> // Для замера времени CPU

// ---------------------------------------------------------
// Часть 1. Функция-ядро (GPU)
// ---------------------------------------------------------
__global__ void function(float *dA, float *dB, float *dC, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    int j;
    float ab, sum = 0.0f;

    if (i < size) {
        ab = dA[i] * dB[i];
        for (j = 0; j < 100; j++) {
            sum = sum + sinf(j + ab);
        }
        dC[i] = sum;
    }
}

int main() {
    // ---------------------------------------------------------
    // Часть 2. Инициализация и выделение памяти
    // ---------------------------------------------------------
    float *hA, *hB, *hC; 
    float *dA, *dB, *dC; 
    float *hC_CPU; // Для результата CPU

    int nStream = 4; 
    int total_N = 512 * 50000;
    int size = total_N / nStream; 
    
    int N_thread = 512;             
    int N_blocks = size / N_thread; 

    unsigned int mem_size = sizeof(float) * size;
    unsigned int total_mem_size = sizeof(float) * total_N;

    // Выделение Pinned Memory (Host)
    cudaMallocHost((void**)&hA, total_mem_size);
    cudaMallocHost((void**)&hB, total_mem_size);
    cudaMallocHost((void**)&hC, total_mem_size);
    hC_CPU = (float*)malloc(total_mem_size);

    // Выделение Device Memory
    cudaMalloc((void**)&dA, total_mem_size);
    cudaMalloc((void**)&dB, total_mem_size);
    cudaMalloc((void**)&dC, total_mem_size);

    // Заполнение данными
    printf("Initializing data...\n");
    for (int i = 0; i < total_N; i++) {
        hA[i] = sinf(i);
        hB[i] = cosf(2.0f * i - 5.0f);
        hC[i] = 0.0f;
        hC_CPU[i] = 0.0f;
    }

    // Создание CUDA-streams
    cudaStream_t stream[4];
    for (int i = 0; i < nStream; ++i) {
        cudaStreamCreate(&stream[i]);
    }

    // Создание событий для тайминга
    cudaEvent_t startTotal, stopTotal;
    cudaEvent_t startKernel, stopKernel;
    cudaEventCreate(&startTotal);
    cudaEventCreate(&stopTotal);
    cudaEventCreate(&startKernel);
    cudaEventCreate(&stopKernel);

    // ---------------------------------------------------------
    // Часть 4. GPU ВАРИАНТ (3 цикла) + Замеры времени
    // ---------------------------------------------------------
    printf("Running GPU calculation...\n");
    
    // Засекаем общее время (включая пересылки)
    cudaEventRecord(startTotal, 0);

    int i; 

    // Цикл 1: Копирование Host -> Device
    for (i = 0; i < nStream; ++i) {
        cudaMemcpyAsync(dA + i * size, hA + i * size, mem_size, 
                        cudaMemcpyHostToDevice, stream[i]);
    }

    // Цикл 2: Запуск ядер
    for (i = 0; i < nStream; ++i) {
        // Засекаем начало работы ядер (в первом стриме)
        if (i == 0) cudaEventRecord(startKernel, stream[i]);

        function<<<N_blocks, N_thread, 0, stream[i]>>>(
            dA + i * size, 
            dB + i * size, 
            dC + i * size, 
            size
        );

        // Засекаем конец работы ядер (в последнем стриме)
        if (i == nStream - 1) cudaEventRecord(stopKernel, stream[i]);
    }

    // Цикл 3: Копирование Device -> Host
    for (i = 0; i < nStream; ++i) {
        cudaMemcpyAsync(hC + i * size, dC + i * size, mem_size, 
                        cudaMemcpyDeviceToHost, stream[i]);
    }

    // Засекаем конец общего времени
    cudaEventRecord(stopTotal, 0);

    // Ждем завершения всего
    cudaDeviceSynchronize();

    // Расчет времени GPU
    float timeTotalMs = 0, timeKernelMs = 0;
    cudaEventElapsedTime(&timeTotalMs, startTotal, stopTotal);
    cudaEventElapsedTime(&timeKernelMs, startKernel, stopKernel);

    // ---------------------------------------------------------
    // Часть 6. CPU ВАРИАНТ (1 поток) + Замер времени
    // ---------------------------------------------------------
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
    // Перевод тиков процессора в миллисекунды
    float cpu_time_ms = 1000.0f * (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC;

    // ---------------------------------------------------------
    // Вывод результатов (как на слайде)
    // ---------------------------------------------------------
    printf("\n------------------------------------------------\n");
    printf("РЕЗУЛЬТАТ\n");
    printf("------------------------------------------------\n");
    
    // Формат: Чистое ядро (С учетом пересылки)
    printf("GPU calculation time: %.0f ms (%.0f)\n", timeKernelMs, timeTotalMs);
    
    printf("CPU calculation time: %.0f ms\n", cpu_time_ms);
    
    // Rate считается относительно общего времени GPU (Total), так как это реальное время ожидания
    float rate = cpu_time_ms / timeTotalMs;
    printf("Rate: %.0f x\n", rate);
    
    printf("CUDA-Streams: %d\n", nStream);
    printf("------------------------------------------------\n");

    // Очистка ресурсов
    for (i = 0; i < nStream; ++i) cudaStreamDestroy(stream[i]);
    cudaEventDestroy(startTotal); cudaEventDestroy(stopTotal);
    cudaEventDestroy(startKernel); cudaEventDestroy(stopKernel);
    
    cudaFreeHost(hA); cudaFreeHost(hB); cudaFreeHost(hC);
    free(hC_CPU);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);

    return 0;
}