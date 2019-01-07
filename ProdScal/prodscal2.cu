#include <stdio.h>
#include <cuda.h>

#include "../cuPrintf.cu"

#define NANOSECONDS_PER_SECOND 1E9;

void initializeArray(int*, int);
void stampaArray(int*, int);
void dotProdCPU(int *, int *, int *, int);
__global__ void dotProdGPU(int*, int*, int*, int);
__global__ void dotProdGPUShared1(int*, int*, int*, int);
__global__ void dotProdGPUShared2(int*, int*, int*, int);
__global__ void reduce1(int*, int*);
__global__ void reduce2(int*, int*);

int main(int argc, char *argv[]) {
	// numero di blocchi e numero di thread per blocco
	dim3 gridDim, blockDim;
    int num; // radice del numero thread del blocco
	int i, N; // numero totale di elementi dell'array
	// array memorizzati sull'host
	int *A_host, *B_host, *C_host;
	// array memorizzati sul device
    int *A_device, *B_device, *C_device;
    int *sharedSums;
	int *copy; // array in cui copieremo i risultati di C_device
	int size, grid_size; // size in byte di ciascun array
    int host_sum, device_local_sum, shared_sum;
    int flag;
    cudaEvent_t startGPU, stopGPU; // tempi di inizio e fine
	struct timespec startCPU, stopCPU;
    float elapsedGPU, elapsedGPUSum, elapsedCPU;
    int numResBlocks;
    int threadPerSM;
    const int NUM_SM = 16; // 16 for Fermi
    const int MAX_NUM_THREADS = 1536; // 1536 for Fermi, 2048 for Kepler
    const int MAX_NUM_BLOCKS = 8; // 8 for Fermi, 16 for Kepler
	const int MS_IN_S = 1000;

    if (argc < 4) {
        printf("Numero di parametri insufficiente!\n");
        printf("Uso corretto: %s <NumElementi> <NumThreadPerBlocco> <flag per la Stampa>\n", argv[0]);
        printf("Uso dei valori di default\n");
        N = 131072;
        num = 32;
        flag = 0;
    }
    else {
        N = atoi(argv[1]);
        num = atoi(argv[2]);
        flag = atoi(argv[3]);
    }

    if (flag) {
        printf("***\t PRODOTTO COMPONENTE PER COMPONENTE DI DUE ARRAY \t***\n");
        printf("Thread per blocco richiesti: %d\nStampa matrici (0 no, 1 sì): %1d\n", N, num, flag);
    }

    printf("Taglia input: %d\n", N);

    blockDim.x = num;
    numResBlocks = MAX_NUM_THREADS / blockDim.x;
    threadPerSM = blockDim.x * MAX_NUM_BLOCKS;
    if (flag) {
        printf("Saranno impiegati %d blocchi di thread.\n", numResBlocks);
        printf("Saranno usati %d streaming multiprocessor su %d.\n", numResBlocks / MAX_NUM_BLOCKS, NUM_SM); // TODO controllare
        if (threadPerSM == MAX_NUM_THREADS) {
            printf("Uso ottimale degli SM!\n");
        }
        else {
            printf("Saranno usati solo %d thread su %d per ogni SM!\n", threadPerSM, MAX_NUM_THREADS);
        }
    }

	// determinazione esatta del numero di blocchi (bilanciamento)
	gridDim.x = N / blockDim.x + ((N % blockDim.x) == 0? 0: 1); // load balancing, punto terzo

    // stampa delle info sull'esecuzione del kernel
    if (flag) {
        printf("Numero di elementi = %d\n", N);
        printf("Numero di thread per blocco = %d\n", blockDim.x);
        printf("Numero di blocchi = %d\n", gridDim.x);
    }

    // allocazione dati sull'host
	size = sizeof(int) * N; // size in byte di ogni array
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

    // chiamata alla funzione seriale per il prodotto di due array
	clock_gettime(CLOCK_REALTIME, &startCPU);
	dotProdCPU(A_host, B_host, C_host, N);
	clock_gettime(CLOCK_REALTIME, &stopCPU);
	elapsedCPU = (stopCPU.tv_sec - startCPU.tv_sec) + (stopCPU.tv_nsec - startCPU.tv_nsec) / NANOSECONDS_PER_SECOND;

    printf("Tempo CPU: %.3f ms\n", elapsedCPU * MS_IN_S);

    printf("\nI STRATEGIA\n");

    // avvia cronometrazione GPU
    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);

    // invocazione del kernel
    cudaEventRecord(startGPU);
    dotProdGPU<<<gridDim, blockDim>>>(A_device, B_device, C_device, N);
    cudaEventRecord(stopGPU);

    // calcola il tempo impiegato dal device per l'esecuzione del kernel
    cudaEventSynchronize(stopGPU);
    cudaEventElapsedTime(&elapsedGPU, startGPU, stopGPU);
    cudaEventDestroy(startGPU);
    cudaEventDestroy(stopGPU);

    // copia dei risultati dal device all'host
	cudaMemcpy(copy, C_device, size, cudaMemcpyDeviceToHost);

    // stampa degli array e dei risultati
	if (flag == 1) {
	 	printf("array A: ");
	 	stampaArray(A_host, N);
	 	printf("\narray B: ");
	 	stampaArray(B_host, N);
	 	printf("\nRisultati host: ");
	 	stampaArray(C_host, N);
	 	printf("\nRisultati device: ");
        stampaArray(copy, N);
        printf("\n");
	}

	// somma gli elementi dei due array
	host_sum = device_local_sum = 0;
	for (i = 0; i < N; i++) {
		host_sum += C_host[i];
    }
	clock_gettime(CLOCK_REALTIME, &startCPU);
	for (i = 0; i < N; i++) {
		device_local_sum += copy[i];
    }
    clock_gettime(CLOCK_REALTIME, &stopCPU);

    // includi il tempo di addizione CPU
    elapsedGPUSum = (stopCPU.tv_sec - startCPU.tv_sec) + (stopCPU.tv_nsec - startCPU.tv_nsec) / NANOSECONDS_PER_SECOND;
    elapsedGPU += (elapsedGPUSum * MS_IN_S);

    // confronta i risultati
    if (flag) {
        printf("La somma sul device (%d) ", device_local_sum);
        if (host_sum != device_local_sum) {
            printf("non ");
        }
        printf("coincide con la somma sull'host (%d)!\n", host_sum);
    }

    printf("Tempo GPU: %.3f\n", elapsedGPU);

    // Seconda strategia in shared memory
    printf("\nII STRATEGIA\n");
    grid_size = sizeof (int) * gridDim.x;

    // azzeriamo il contenuto della matrice C e della variabile somma
    cudaFree(C_device);
    cudaMalloc((void **) &C_device, grid_size);
    cudaMemset(C_device, 0, grid_size);

    // allochiamo il vettore delle somme parziali
    sharedSums = (int *) calloc(N, sizeof(int));

    // avvia cronometrazione GPU
    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);

    // invocazione del kernel
    if (flag) {
        cudaPrintfInit();
    }
    cudaEventRecord(startGPU);
    // dotProdGPU<<<gridDim, blockDim>>>(A_device, B_device, C_device, N); // prodotto
    // reduce1<<<gridDim, blockDim, blockDim.x * sizeof(int)>>>(C_device, C_device); // somma 2 strategia
    dotProdGPUShared1<<<gridDim, blockDim, blockDim.x * sizeof(int)>>>(A_device, B_device, C_device, N);
    cudaEventRecord(stopGPU);
    if (flag) {
        cudaPrintfDisplay();
        cudaPrintfEnd();
    }

    // calcola il tempo impiegato dal device per l'esecuzione del kernel
    cudaEventSynchronize(stopGPU);
    cudaEventElapsedTime(&elapsedGPU, startGPU, stopGPU);
    cudaEventDestroy(startGPU);
    cudaEventDestroy(stopGPU);

    // copia dei risultati dal device all'host
	cudaMemcpy(sharedSums, C_device, grid_size, cudaMemcpyDeviceToHost);

    // stampa degli array e dei risultati
    if (flag == 1) {
        printf("array A: ");
        stampaArray(A_host, N);
        printf("\narray B: ");
        stampaArray(B_host, N);
        printf("\nRisultati host: ");
        stampaArray(C_host, N);
        printf("\nRisultati device: ");
        stampaArray(sharedSums, N);
        printf("\n");
    }

    // somma gli elementi dei due array
    shared_sum = 0;
    for (i = 0; i < gridDim.x; i++) {
        // printf("sharedSums[%d] = %d\n", i, sharedSums[i]);
        shared_sum += sharedSums[i];
    }

    // confronta i risultati
    if (flag) {
        printf("La somma sul device (%d) ", shared_sum);
        if (host_sum != shared_sum) {
            printf("non ");
        }
        printf("coincide con la somma sull'host (%d)!\n", host_sum);
    }

    printf("Tempo GPU: %.3f\n", elapsedGPU);

    // terza strategia
    printf("\nIII STRATEGIA\n");

    // azzeriamo il contenuto della matrice C e della variabile somma
    cudaMemset(C_device, 0, grid_size);

    // azzeriamo il vettore delle somme parziali
    memset(sharedSums, 0, size);

    // avvia cronometrazione GPU
    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);

    // invocazione del kernel
    cudaEventRecord(startGPU);
    // dotProdGPU<<<gridDim, blockDim>>>(A_device, B_device, C_device, N); // prodotto
    // reduce2<<<gridDim, blockDim, blockDim.x * sizeof(int)>>>(C_device, C_device); // somma 3 strategia
    dotProdGPUShared2<<<gridDim, blockDim, blockDim.x * sizeof(int)>>>(A_device, B_device, C_device, N);
    cudaEventRecord(stopGPU);

    // calcola il tempo impiegato dal device per l'esecuzione del kernel
    cudaEventSynchronize(stopGPU);
    cudaEventElapsedTime(&elapsedGPU, startGPU, stopGPU);
    cudaEventDestroy(startGPU);
    cudaEventDestroy(stopGPU);

    // copia dei risultati dal device all'host
	cudaMemcpy(sharedSums, C_device, grid_size, cudaMemcpyDeviceToHost);

    // stampa degli array e dei risultati
	if (flag == 1) {
        printf("array A: ");
        stampaArray(A_host, N);
        printf("\narray B: ");
        stampaArray(B_host, N);
        printf("\nRisultati host: ");
        stampaArray(C_host, N);
        printf("\nRisultati device: ");
        stampaArray(sharedSums, N);
        printf("\n");
    }

	// somma gli elementi dei due array
    shared_sum = 0;
    for (i = 0; i < gridDim.x; i++) {
        // printf("sharedSums[%d] = %d\n", i, sharedSums[i]);
        shared_sum += sharedSums[i];
    }

    // confronta i risultati
    if (flag) {
        printf("La somma sul device (%d) ", shared_sum);
        if (host_sum != shared_sum) {
            printf("non ");
        }
        printf("coincide con la somma sull'host (%d)!\n", host_sum);
    }

    printf("Tempo GPU: %.3f\n", elapsedGPU);

    // de-allocazione host
	free(A_host);
	free(B_host);
	free(C_host);
    free(copy);
    free(sharedSums);

	// de-allocazione device
	cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);

    if (flag) {
        printf("\nFine programma %s.\n", argv[0]);
    }

    return EXIT_SUCCESS;
}

void initializeArray(int *array, int n) {
	int i;

	for (i = 0; i < n; i++)
        array[i] = rand() % 5;
}

void stampaArray(int* array, int n) {
	int i;

	for (i = 0; i < n; i++)
		printf("%d ", array[i]);
}

// Seriale
void dotProdCPU(int *a, int *b, int *c, int n) {
	int i;

	for (i = 0; i < n; i++)
		c[i] = a[i] * b[i];
}

// Parallelo
__global__ void dotProdGPU(int *a, int *b, int *c, int n) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < n)
		c[index] = a[index] * b[index];
}

__global__ void dotProdGPUShared(int *a, int *b, int n) {
    extern __shared__ int sdata[];
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n)
        sdata[threadIdx.x] = a[index] + b[index];
}

__global__ void dotProdGPUShared1(int *a, int *b, int *c, int n) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

    // create vector product
    if (index < n)
        sdata[threadIdx.x] = a[index] * b[index];
    __syncthreads();

    // print stared data array (for debug purposes)
    // if (tid == 0) {
    //     for (unsigned int i = 0; i < blockDim.x; i++) {
    //         cuPrintf("[T%d] sdata[%d] = %d\n", tid, i, sdata[i]);
    //     }
    // }

    // do reduction in shared mem
    for (unsigned int step = 1; step < blockDim.x; step *= 2) { 
        // step = x*2
        if (tid % (2 * step) == 0) { // only threadIDs divisible by step participate
            sdata[tid] += sdata[tid + step];
            // cuPrintf("[T%d] Step %d, sdata[%d] = %d\n", tid, step, tid, sdata[tid]);
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) {
        c[blockIdx.x] = sdata[tid]; // tid is 0 anyway
        // cuPrintf("[T%d] local sum: %d\n", tid, c[blockIdx.x]);
    }
}

__global__ void dotProdGPUShared2(int *a, int *b, int *c, int n) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

    // create vector product
    if (index < n)
        sdata[threadIdx.x] = a[index] * b[index];
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int step = blockDim.x/2; step > 0; step >>= 1) { 
        // s = s/2
        if (threadIdx.x < step) { 
            sdata[tid] += sdata[threadIdx.x + step];
        }

    __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) {
        c[blockIdx.x] = sdata[0]; // tid is 0 anyway
    }
}

__global__ void reduce1(int *g_idata, int *g_odata) {
    extern __shared__ int sdata[];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[index];
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s = 1; s < blockDim.x; s *= 2) { 
        // step = x*2
        if (tid % (2*s) == 0) { // only threadIDs divisible by step participate
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[tid]; // tid is 0 anyway
}

__global__ void reduce2(int *g_idata, int *g_odata) {
    extern __shared__ int sdata[];
  
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[index];
    __syncthreads();
  
    // do reduction in shared mem
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) { 
        // s = s/2
        if (tid < s) { 
            sdata[tid] += sdata[tid + s];
        }

    __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) 
        g_odata[blockIdx.x] = sdata[tid]; // tid is 0 anyway
}
