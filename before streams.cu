//Problem 6 Solution

#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>



__global__
void perfectNumbersKernel(int n, int hardLimit)//, int* d_perfectArray)
{


    int blockId = blockIdx.y * gridDim.x + blockIdx.x + blockIdx.z * gridDim.x * gridDim.y;
    int currentNum = blockId * blockDim.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.x + threadIdx.x + threadIdx.z * blockDim.x * blockDim.y;
    

    if (currentNum < hardLimit)
    {
        int sum = 0;

        for (int i = 1; i < currentNum; i++)
        {
            if (currentNum % i == 0)
                sum = sum + i;
        }
        if (currentNum == sum)
        {
            //d_perfectArray[counter - 1] = currentNum;           //potential race condition here
            //counter++;
            printf("%d  ", currentNum);
        }
    }

}


void perfectNumbers(int n, int hardLimit)//, int * perfectArray)
{
    //int size = n * sizeof(int);
    //int* d_perfectArray;
    //cudaMalloc((void**)&d_perfectArray, size);
    //cudaMemcpy(d_perfectArray, perfectArray, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(1024, 512, 32);                
    dim3 dimBlock(128, 4, 2);                   

    perfectNumbersKernel <<< dimGrid, dimBlock >>> (n, hardLimit);// , d_perfectArray);

  /* From device query ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\extras\demo_suite\deviceQuery.exe")
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  */
}



int main()
{

    int n = 1;
    int counter = 1;
    int hardLimit = 0;

    printf("A perfect number is a positive integer whose value is equal to the sum of all of its positive factors,excluding itself.\n");
    printf("This program will list the first N perfect numbers\n\n");
    printf("Please specify N: ");
    scanf_s("%d", &n);

    printf("Specify upper a hard limit for calculation: ");
    scanf_s("%d", &hardLimit);

   // int* perfectArray;
    //perfectArray = (int*)malloc(n * sizeof(int));

   // for (int i = 0; i < n; i++)
    //    perfectArray[i] = 0;


    perfectNumbers(n, hardLimit);//, perfectArray);



    //for (int i = 0; i < n; i++)
    {
       // printf("%d ", perfectArray[i]);
    }


    //free(perfectArray);

    return 0;
}


/*
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}

*/