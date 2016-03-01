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
    int N = 3000;
    int SIZE = N*N;
	cudaEvent_t start, stop;
 	float elapsedTime;
    // Allocate memory on the host
    vector<double> h_A(SIZE);
    vector<double> h_B(SIZE);
    vector<double> h_C(SIZE);

    // Initialize matrices on the host
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            h_A[i*N+j] = sin(i);
            h_B[i*N+j] = cos(j);
        }
    }

    // Allocate memory on the device
    dev_array<double> d_A(SIZE);
    dev_array<double> d_B(SIZE);
    dev_array<double> d_C(SIZE);

    d_A.set(&h_A[0], SIZE);
    d_B.set(&h_B[0], SIZE);
	cudaEventCreate( &start );
 	cudaEventCreate( &stop );
 	cudaEventRecord( start, 0 );

    matrixMultiplication(d_A.getData(), d_B.getData(), d_C.getData(), N);
    cudaDeviceSynchronize();

	cudaEventRecord( stop, 0 );
 	cudaEventSynchronize( stop );
 	cudaEventElapsedTime( &elapsedTime, start, stop );

	cout << "Time taken: "<< elapsedTime << "ms" << endl;

    d_C.get(&h_C[0], SIZE);
    cudaDeviceSynchronize();


    return 0;
}
