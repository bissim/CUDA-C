#include <cuda.h>
#include <stdio.h>

void initializeArray(int*, int);
void stampaArray(int*, int);
void equalArray(int*, int*, int);
void prodottoArrayCompPerCompCPU(int*, int*, int *, int);
__global__ void prodottoArrayCompPerCompGPU(int*, int*, int*, int );

int main(int argn, char * argv[]) {
    //numero di blocchi e numero di thread per blocco
    dim3 nBlocchi, nThreadPerBlocco;
    int N; //numero totale di elementi dell'array
    //array memorizzati sull'host
    int *A_host, *B_host, *C_host;
    //array memorizzati sul device
    int *A_device, *B_device, *C_device;
    int *copy; //array in cui copieremo i risultati di C_device
    int size; //size in byte di ciascun array
    int flag;

    printf("***\t PRODOTTO COMPONENTE PER COMPONENTE DI DUE ARRAY \t***\n");
    /* se l'utente non ha inserito un numero sufficiente di
    parametri da riga di comando, si ricorre ai valori di
    default per impostare il numero di thread per blocco, il
    numero totale di elementi e il flag di stampa */
    if (argn<4) {
        printf("Numero di parametri insufficiente!!!\n");
        printf("Uso corretto: %s <NumElementi> <NumThreadPerBlocco> <flag per la Stampa>\n",argv[0]);
        printf("Uso dei valori di default\n");
        nThreadPerBlocco=4;
        N=12;
        flag=1;
    } else {
        N=atoi(argv[1]);
        nThreadPerBlocco=atoi(argv[2]);
        flag=atoi(argv[3]);
    }

    //determinazione esatta del numero di blocchi
    nBlocchi = N / nThreadPerBlocco.x + ((N%nThreadPerBlocco.x)==0?0:1);
    //size in byte di ogni array
    size = N * sizeof(int);

    //stampa delle info sull'esecuzione del kernel
    printf("Numero di elementi = %d\n", N);
    printf("Numero di thread per blocco = %d\n", nThreadPerBlocco.x);
    printf("Numero di blocchi = %d\n", nBlocchi.x);

    //allocazione dati sull'host
    A_host = (int*)malloc(size);
    B_host = (int*)malloc(size);
    C_host = (int*)malloc(size);
    copy = (int*)malloc(size);

    //allocazione dati sul device
    cudaMalloc((void**) &A_device, size);
    cudaMalloc((void**) &B_device, size);
    cudaMalloc((void**) &C_device, size);

    //inizializzazione dati sull'host
    initializeArray(A_host, N);
    initializeArray(B_host, N);

    //copia dei dati dall'host al device
    cudaMemcpy(A_device, A_host, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B_host, size, cudaMemcpyHostToDevice);

    //azzeriamo il contenuto della matrice C
    memset(C_host, 0, size);
    cudaMemset(C_device, 0, size);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    //invocazione del kernel
    prodottoArrayCompPerCompGPU<<<nBlocchi, nThreadPerBlocco>>>(A_device, B_device, C_device, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop); // assicura che tutti siano arrivati all'evento stop prima di registrare il tempo
    float elapsed;
    // tempo tra i due eventi in millisecondi
    cudaEventElapsedTime(&elapsed, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    //copia dei risultati dal device all'host
    cudaMemcpy(copy,C_device,size, cudaMemcpyDeviceToHost);

    printf("tempo GPU: %.3f\n", elapsed);

    // calcolo su CPU
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    //chiamata alla funzione seriale per il prodotto di due array
    prodottoArrayCompPerCompCPU(A_host, B_host, C_host, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop); // assicura che tutti siano arrivati all'evento stop prima di registrare il tempo
    cudaEventElapsedTime(&elapsed, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("tempo CPU: %.3f\n", elapsed);


    //stampa degli array e dei risultati
    if (flag == 1) {
        printf("array A\n");
        stampaArray(A_host,N);
        printf("array B\n");
        stampaArray(B_host,N);
        printf("Risultati host\n");
        stampaArray(C_host, N);
        printf("Risultati device\n");
        stampaArray(copy,N);
    }

    //test di correttezza
    equalArray(copy, C_host,N);

    //de-allocazione host
    free(A_host);
    free(B_host);
    free(C_host);
    free(copy);

    //de-allocazione device
    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);
    exit(0);
}

void initializeArray(int *array, int n) {
    int i;

    for (i = 0; i < n; i++)
        array[i] = 1/((i+1)*10);
        if (i % 2 == 0)
            array[i] = array[i]*(-1);
}

void stampaArray(int* array, int n) {
    int i;

    for (i = 0; i < n; i++)
        printf("%d ", array[i]);
        printf("\n");
}

void equalArray(int* a, int* b, int n) {
    int i = 0;

    while (a[i] == b[i])
        i++;
    if (i < n)
        printf("I risultati dell'host e del device sono diversi\n");
    else
        printf("I risultati dell'host e del device coincidono\n");
}

//Seriale
void prodottoArrayCompPerCompCPU(int *a, int *b, int *c, int n) {
    int i;

    for (i = 0; i < n; i++)
        c[i]=a[i]*b[i];
}

//Parallelo
__global__ void prodottoArrayCompPerCompGPU(int *a, int *b, int *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < n)
        c[index] = a[index]*b[index];
}
