#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "kernel.h"
#include "dev_array.h"
#include <math.h>

using namespace std;

int main()
{
    // Perform matrix multiplication C = A*B
    // where A, B and C are NxN matrices
    int N = 2000;
    int SIZE = N*N;
	cudaEvent_t start, stop;
 	float elapsedTime;
    // Allocate memory on the host
    vector<float> h_A(SIZE);
    vector<float> h_B(SIZE);
    vector<float> h_C(SIZE);

    // Initialize matrices on the host
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            h_A[i*N+j] = sin(i);
            h_B[i*N+j] = cos(j);
        }
    }

    // Allocate memory on the device
    dev_array<float> d_A(SIZE);
    dev_array<float> d_B(SIZE);
    dev_array<float> d_C(SIZE);

    d_A.set(&h_A[0], SIZE);
    d_B.set(&h_B[0], SIZE);
	HANDLE_ERROR( cudaEventCreate( &start ) );
 	HANDLE_ERROR( cudaEventCreate( &stop ) );
 	HANDLE_ERROR( cudaEventRecord( start, 0 ) );

    matrixMultiplication(d_A.getData(), d_B.getData(), d_C.getData(), N);
    cudaDeviceSynchronize();

	HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
 	HANDLE_ERROR( cudaEventSynchronize( stop ) );
 	HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );

	printf( "Time taken: %3.1f ms\n", elapsedTime );

    d_C.get(&h_C[0], SIZE);
    cudaDeviceSynchronize();


    return 0;
}
