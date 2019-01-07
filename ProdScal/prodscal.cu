#include <stdio.h>
#include <cuda.h>
#include <time.h>

#include "../cuPrintf.cu"

#define NANOSECONDS_PER_SECOND 1E9;

void initializeArray(int*, int);
void stampaArray(int*, int);
void equalArray(int*, int*, int);
void prodottoArrayCompPerCompCPU(int *, int *, int *, int);
__global__ void prodottoArrayCompPerCompGPU(int*, int*, int*, int);

int main(int argc, char *argv[]) {
	// numero di blocchi e numero di thread per blocco
	dim3 gridDim, blockDim;
	int i, N; //numero totale di elementi dell'array
	// array memorizzati sull'host
	int *A_host, *B_host, *C_host;
	// array memorizzati sul device
	int *A_device, *B_device, *C_device;
	int *copy; //array in cui copieremo i risultati di C_device
	int size; //size in byte di ciascun array
	int num;
	int host_sum, device_sum;
    int flag, errorFlag;
    cudaEvent_t startGPU, stopGPU; // tempi di inizio e fine
    float elapsedGPU, elapsedCPU;
	struct timespec startCPU, stopCPU;
	// const long NANOSECONDS_PER_SECOND = 1E9;
	const int MS_IN_S = 1000;

	if (argc < 4) {
        printf("Numero di parametri insufficiente!\n");
        printf("Uso corretto: %s <NumElementi> <NumThreadPerBlocco> <debugFlag>\n", argv[0]);
        printf("Uso dei valori di default\n");
        N = 128;
        num = 32;
        flag = 0;
    }
    else {
        N = atoi(argv[1]);
        num = atoi(argv[2]);
        flag = atoi(argv[3]);
    }
	blockDim.x = num; // it should be 32 this time

	// determinazione esatta del numero di blocchi
	gridDim.x = N / blockDim.x + ((N % blockDim.x) == 0? 0: 1); // load balancing, punto terzo

	// size in byte di ogni array
	size = sizeof(int) * N;

	// stampa delle info sull'esecuzione del kernel
	if (flag) {
		printf("***\t PRODOTTO COMPONENTE PER COMPONENTE DI DUE ARRAY \t***\n");
		printf("Numero di elementi = %d\n", N);
		printf("Numero di thread per blocco = %d\n", blockDim.x);
		printf("Numero di blocchi = %d\n", gridDim.x);
	}

	// allocazione dati sull'host
	A_host = (int *) malloc(size);
	B_host = (int *) malloc(size);
	C_host = (int *) malloc(size);
	copy = (int *) malloc(size);

	// allocazione dati sul device
	cudaMalloc((void **) &A_device, size);
	cudaMalloc((void **) &B_device, size);
	cudaMalloc((void **) &C_device, size);

	// inizializzazione dati sull'host
	initializeArray(A_host, N);
	initializeArray(B_host, N);

	// copia dei dati dall'host al device
	cudaMemcpy(A_device, A_host, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_device, B_host, size, cudaMemcpyHostToDevice);

	// azzeriamo il contenuto della matrice C
	memset(C_host, 0, size);
	cudaMemset(C_device, 0, size);

    // avvia cronometrazione GPU
    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);

	// invocazione del kernel
	cudaEventRecord(startGPU);
	prodottoArrayCompPerCompGPU<<<gridDim, blockDim>>>(A_device, B_device, C_device, N);
	cudaEventRecord(stopGPU);

    // calcola il tempo impiegato dal device per l'esecuzione del kernel
    cudaEventSynchronize(stopGPU);
    cudaEventElapsedTime(&elapsedGPU, startGPU, stopGPU);
    cudaEventDestroy(startGPU);
    cudaEventDestroy(stopGPU);

	// copia dei risultati dal device all'host
	cudaMemcpy(copy, C_device, size, cudaMemcpyDeviceToHost);

	// chiamata alla funzione seriale per il prodotto di due array
	clock_gettime(CLOCK_REALTIME, &startCPU);
	prodottoArrayCompPerCompCPU(A_host, B_host, C_host, N);
	clock_gettime(CLOCK_REALTIME, &stopCPU);
	elapsedCPU = (stopCPU.tv_sec - startCPU.tv_sec) + (stopCPU.tv_nsec - startCPU.tv_nsec) / NANOSECONDS_PER_SECOND;

	// stampa degli array e dei risultati
	if (flag && N < 20) {
	 	printf("array A\n");
		stampaArray(A_host, N);
		printf("array B\n");
		stampaArray(B_host, N);
		printf("Risultati host\n");
		stampaArray(C_host, N);
		printf("Risultati device\n");
		stampaArray(copy,N);
	}

	// test di correttezza
	if (flag) { 
		equalArray(copy, C_host, N);
	}

	// somma gli elementi dei due array
	host_sum = device_sum = 0;
	for (i = 0; i < N; i++) {
		host_sum += C_host[i];
		device_sum += copy[i];
	}

	// confronta i risultati
	errorFlag = 0;
	if (flag) {
		printf("La somma sul device (%d) ", device_sum);
		if (host_sum != device_sum) {
			printf("non ");
			errorFlag = 1;
		}
		printf("coincide con la somma sull'host (%d)!\n", host_sum);
	}
	else if (errorFlag) {
		printf("Le somme non coincidono!");
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

	exit(EXIT_SUCCESS);
}

void initializeArray(int *array, int n) {
	int i;

	for (i = 0; i < n; i++)
		array[i] = i;
}

void stampaArray(int* array, int n) {
	int i;

	for (i = 0; i < n; i++)
		printf("%d ", array[i]);
	printf("\n");
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

// Seriale
void prodottoArrayCompPerCompCPU(int *a, int *b, int *c, int n) {
	int i;

	for (i = 0; i < n; i++)
		c[i] = a[i] * b[i];
}

// Parallelo
__global__ void prodottoArrayCompPerCompGPU(int *a, int *b, int *c, int n) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < n)
		c[index] = a[index] * b[index];
}
