#include <cuda.h>
#include <stdio.h>

void initializeArray(int*, int);
void stampaMatriceArray(int*, int, int);
void equalArray(int*, int*, int);
void sommaMatriciCPU(int*, int*, int*, int, int);
__global__ void sommaMatriciGPU(int*, int*, int*, int, int);

int main(int argc, char **argv) {
	// numero di blocchi e numero di thread per blocco
	dim3 gridDim, blockDim(4, 8);
    int N; //numero totale di elementi dell'array
	// array memorizzati sull'host
	int *A_host, *B_host, *C_host;
	// array memorizzati sul device
	int *A_device, *B_device, *C_device;
	int *copy; // array in cui copieremo i risultati di C_device
	int size; // size in byte di ciascun array
    cudaEvent_t start, stop; // tempi di inizio e fine

    printf("Addizione di due matrici quadrate.\n");
    printf("Inserire n: ");
    scanf("%d", &N);

    // determinazione esatta del numero di blocchi
    gridDim.x = N / blockDim.x + ((N % blockDim.x) == 0? 0: 1);
    gridDim.y = N / blockDim.y + ((N % blockDim.y) == 0? 0: 1);
    size = N * N * sizeof(int); // dimensione in byte delle matrici
	// stampa delle info sull'esecuzione del kernel
	printf("Numero di elementi = %d\n", N);
	printf("Numero di thread per blocco = %d\n", blockDim.x * blockDim.y);
	printf("Numero di blocchi = %d\n", gridDim.x * gridDim.y);

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
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // invocazione del kernel
    sommaMatriciGPU<<<gridDim, blockDim>>>(A_device, B_device, C_device, N, N);

    // ferma cronometrazione GPU
    cudaEventRecord(stop);
    cudaEventSynchronize(stop); // assicura che tutti siano arrivati all'evento stop prima di registrare il tempo
    float elapsed;
    // tempo tra i due eventi in millisecondi
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("tempo GPU: %f\n", elapsed);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // copia dei risultati dal device all'host
	cudaMemcpy(copy, C_device, size, cudaMemcpyDeviceToHost);

    // avvia cronometrazione CPU
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // invocazione somma seriale
    sommaMatriciCPU(A_host, B_host, C_host, N, N);

    // arresta cronometrazione CPU
    cudaEventRecord(stop);
    cudaEventSynchronize(stop); // assicura che tutti siano arrivati all'evento stop prima di registrare il tempo
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("tempo CPU: %f\n", elapsed);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // test di correttezza
    equalArray(copy, C_host, N * N);

	// de-allocazione host
	free(A_host);
	free(B_host);
	free(C_host);
	free(copy);

	// de-allocazione device
	cudaFree(A_device);
	cudaFree(B_device);
	cudaFree(C_device);

    return 0;
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

void equalArray(int* a, int*b, int n) {
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

    if (index < rows*cols) {
        result[index] = first[index] + second[index]; 
    }
}
