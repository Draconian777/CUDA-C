//Issa Jawabreh - OAFE3F
//Parallel Technologies 2 - Problem 6 Solution

#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>

#define NSTREAM 32

__global__
void perfectNumbersKernel(int start, int upperLimit, int streamID, int sizeOfDataBlock)
 {
 


    int begin = streamID * sizeOfDataBlock + start;
    int end = begin + sizeOfDataBlock;
    
    
    //printf("\nStream %d parsing from %d to %d", streamID, begin, end);


    int blockId = blockIdx.y * gridDim.x + blockIdx.x + blockIdx.z * gridDim.x * gridDim.y;
    int currentNum = blockId * blockDim.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.x + threadIdx.x + threadIdx.z * blockDim.x * blockDim.y;

    

    if (currentNum < end && currentNum>begin)
    {
        int sum = 0;

        for (int i = 1; i < currentNum; i++)
        {
            if (currentNum % i == 0)
                sum = sum + i;
        }
        if (currentNum == sum)
        {

            printf("\nStream %d found %d ",streamID, currentNum);
        }
    }

}


#if defined(WIN32)
int setenv(const char* name, const char* value, int overwrite)
{
    int errcode = 0;
    if (!overwrite) {
        size_t envsize = 0;
        errcode = getenv_s(&envsize, NULL, 0, name);
        if (errcode || envsize) return errcode;
    }
    return _putenv_s(name, value);
}
#endif


void perfectNumbers(int start, int upperLimit)//, int * perfectArray)
{
    //int size = n * sizeof(int);
    //int* d_perfectArray;
    //cudaMalloc((void**)&d_perfectArray, size);
    //cudaMemcpy(d_perfectArray, perfectArray, size, cudaMemcpyHostToDevice);

/*
    dim3 dimGrid(1024, 512, 32);
    dim3 dimBlock(128, 1, 1);
*/
    dim3 dimGrid(1024,1024,1);
    dim3 dimBlock(400);


    cudaStream_t stream[NSTREAM];


    //create non-blocking streams
    for (int i = 0; i < NSTREAM; i++)
    {
        cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
    }


    //non-blocking test
    //cudaError_t cudaStreamQuery ( cudaStream_t stream );
    
    //start= dBlockIndex * sizeOfDataBlock + start + i
    //

    //divide the data into working blocks based on the number of streams
    int sizeOfDataBlock = (upperLimit - start) / NSTREAM;
    int remainder = (upperLimit - start) % NSTREAM;
     

    //for (int dBlockIndex = 0; dBlockIndex < sizeOfDataBlock; dBlockIndex++)
    {
        for (int i = 0; i < NSTREAM; i++) 
        {
            perfectNumbersKernel <<< dimGrid, dimBlock, 0, stream[i] >>> (start, upperLimit, i, sizeOfDataBlock);
            //perfectNumbersKernel <<< dimGrid, dimBlock, 0, stream[i] >> > (start + i * sizeOfDataBlock , start+i * sizeOfDataBlock + sizeOfDataBlock, i, sizeOfDataBlock);
            //perfectNumbersKernel <<< dimGrid, dimBlock, 0, stream[i] >>> (dBlockIndex * sizeOfDataBlock + start + i);
        }         
    }
    
   
    //handle the last bit of data if it does not divide evenly
  /*
    if (remainder)
    {
       
        for (int i = 0; i < remainder; i++)
        {
            perfectNumbersKernel <<< dimGrid, dimBlock, 0, stream[i] >>> (sizeOfDataBlock* NSTREAM + i);
        }
    }
    */

    //perfectNumbersKernel <<< dimGrid, dimBlock >>> (start, upperLimit);// , d_perfectArray);
    
    //Destroy Streams
    for (int i = 0; i < NSTREAM; i++)
    {
        cudaStreamDestroy(stream[i]);
    }

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
    int start = 2; // to exclude 0 and 1 from results
    int upperLimit = 10000;


    // set up max connection
    char* iname = "CUDA_DEVICE_MAX_CONNECTIONS";
    setenv(iname, "32", 1);
    char* ivalue = getenv(iname);
    printf("> %s = %s\n", iname, ivalue);
    printf("> with streams = %d\n", NSTREAM);



    printf("A perfect number is a positive integer whose value is equal to the sum of all of its positive factors,excluding itself.\n");
    printf("This program will find the perfect numbers up to an upper limit specified by the user.\n");
    /*
    printf("This program will list the first N perfect numbers\n\n");
    printf("Please specify N: ");
    scanf_s("%d", &n);
    
    */

    printf("Specify upper an upper limit for calculation: ");
    scanf_s("%d", &upperLimit);

   // int* perfectArray;
    //perfectArray = (int*)malloc(n * sizeof(int));

   // for (int i = 0; i < n; i++)
    //    perfectArray[i] = 0;


    int sizeOfDataBlock = (upperLimit - start) / NSTREAM;
    int remainder = (upperLimit - start) % NSTREAM;

    printf("size of datablock is %d  \n", sizeOfDataBlock);
    printf("remainder is %d  \n\n\n", remainder);


    printf("Perfect Numbers are: ");

    perfectNumbers(start, upperLimit);//, perfectArray);



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