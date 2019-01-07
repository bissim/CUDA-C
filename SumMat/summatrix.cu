#include <stdio.h>
#include <cuda.h>
#include <time.h>

#include "../cuPrintf.cu"

#define NANOSECONDS_PER_SECOND 1E9;

void initializeArray(int*, int);
void stampaMatriceArray(int*, int, int);
void equalArray(int*, int*, int);
void sommaMatriciCPU(int*, int*, int*, int, int);
__global__ void sommaMatriciGPU(int*, int*, int*, int, int);

int main(int argc, char **argv) {
	// numero di blocchi e numero di thread per blocco
	dim3 gridDim, blockDim(4, 8); // blocchi di 32 thread
    int N; //numero totale di elementi dell'array
	// array memorizzati sull'host
	int *A_host, *B_host, *C_host;
	// array memorizzati sul device
	int *A_device, *B_device, *C_device;
	int *copy; // array in cui copieremo i risultati di C_device
	int size; // size in byte di ciascun array
    cudaEvent_t startGPU, stopGPU; // tempi di inizio e fine
    float elapsedGPU, elapsedCPU;
	struct timespec startCPU, stopCPU;
    const int MS_IN_S = 1000;
    int flag;

	if (argc < 3) {
        printf("Numero di parametri insufficiente!\n");
        printf("Uso corretto: %s <NumElementi> <debugFlag>\n", argv[0]);
        printf("Uso dei valori di default\n");
        N = 256;
        flag = 0;
    }
    else {
        N = atoi(argv[1]);
        flag = atoi(argv[2]);
    }

    if (flag) {
        printf("Addizione di due matrici quadrate.\n");
    }

    // determinazione esatta del numero di blocchi
    gridDim.x = N / blockDim.x + ((N % blockDim.x) == 0? 0: 1);
    gridDim.y = N / blockDim.y + ((N % blockDim.y) == 0? 0: 1);

    size = sizeof(int) * N * N; // dimensione in byte delle matrici

    // stampa delle info sull'esecuzione del kernel
    if (flag) {
        printf("Numero di elementi = %d\n", N);
        printf("Numero di thread per blocco = %d\n", blockDim.x * blockDim.y);
        printf("Numero di blocchi = %d\n", gridDim.x * gridDim.y);
    }

    // allocazione dati sull'host
	A_host = (int*) malloc(size);
	B_host = (int*) malloc(size);
	C_host = (int*) malloc(size);
	copy = (int*) malloc(size);

    // allocazione dati sul device
	cudaMalloc((void**) &A_device, size);
	cudaMalloc((void**) &B_device, size);
	cudaMalloc((void**) &C_device, size);

    // inizializzazione dati sull'host
	initializeArray(A_host, N * N);
	initializeArray(B_host, N * N);

    // copia dei dati dall'host al device
	cudaMemcpy(A_device, A_host, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_device, B_host, size, cudaMemcpyHostToDevice);

    // azzeriamo il contenuto della matrice C
	memset(C_host, 0, size);
	cudaMemset(C_device, 0, size);

    // avvia cronometrazione GPU
    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);
    cudaEventRecord(startGPU);

    // invocazione del kernel
    sommaMatriciGPU<<<gridDim, blockDim>>>(A_device, B_device, C_device, N, N);

    // ferma cronometrazione GPU
    cudaEventRecord(stopGPU);
    cudaEventSynchronize(stopGPU); // assicura che tutti siano arrivati all'evento stop prima di registrare il tempo

    // tempo tra i due eventi in millisecondi
    cudaEventElapsedTime(&elapsedGPU, startGPU, stopGPU);
    cudaEventDestroy(startGPU);
    cudaEventDestroy(stopGPU);

    // copia dei risultati dal device all'host
	cudaMemcpy(copy, C_device, size, cudaMemcpyDeviceToHost);

    // invocazione somma seriale
	clock_gettime(CLOCK_REALTIME, &startCPU);
    sommaMatriciCPU(A_host, B_host, C_host, N, N);
	clock_gettime(CLOCK_REALTIME, &stopCPU);
	elapsedCPU = (stopCPU.tv_sec - startCPU.tv_sec) + (stopCPU.tv_nsec - startCPU.tv_nsec) / NANOSECONDS_PER_SECOND;

    // test di correttezza
    if (flag) {
        equalArray(copy, C_host, N * N);
    }

    printf("Tempo CPU: %.3f ms\n", elapsedCPU * MS_IN_S);
    printf("Tempo GPU: %.3f ms\n", elapsedGPU); // already in ms

	// de-allocazione host
	free(A_host);
	free(B_host);
	free(C_host);
	free(copy);

	// de-allocazione device
	cudaFree(A_device);
	cudaFree(B_device);
	cudaFree(C_device);

    return EXIT_SUCCESS;
}

void initializeArray(int *array, int n) {
	int i;

	for (i = 0; i < n; i++)
		array[i] = i;
}

void stampaMatriceArray(int* array, int rows, int cols) {
	int i;

	for (i = 0; i < rows * cols; i++) {
		printf("%6.3d\t", array[i]);

        if (i % cols == cols - 1) {
            printf("\n");
        }
    }
}

void equalArray(int *a, int *b, int n) {
	int i = 0;
	while (a[i] == b[i])
		i++;
	if (i < n)
		printf("I risultati dell'host e del device sono diversi\n");
	else
		printf("I risultati dell'host e del device coincidono\n");
}

// seriale
void sommaMatriciCPU(int *first, int *second, int *result, int rows, int cols) {
    int i;

    for (i = 0; i < rows*cols; i++) {
        result[i] = first[i] + second[i];
    }
}

// parallelo
__global__ void sommaMatriciGPU(int *first, int *second, int *result, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int index = j * gridDim.x * blockDim.x + i;

    if (index < rows * cols) {
        result[index] = first[index] + second[index]; 
    }
}
