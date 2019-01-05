#include<cuda.h>
#include<stdio.h>
void initializeArray(int*,int);
void stampaArray(int*, int);
void equalArray(int*, int*, int);
int main(int argn, char * argv[])
{ 
//numero totale di elementi dell'array
int N;
int *A_host; //array memorizzato sull'host
int *A_device; //array memorizzato sul device
int *copy; //array in cui copieremo i dati dal device
int size; //size in byte di ciascun array
if(argn==1)
N=20;
else
N=atoi(argv[1]);
printf("**********\tPROGRAMMA INIZIALE\t**********\n");
printf("copia di %d elementi dalla CPU alla GPU e viceversa\n\n", N);
//size in byte di ogni array
size=N*sizeof(int);
//allocazione dati sull'host
A_host=(int*)malloc(size);
copy=(int*)malloc(size);
//allocazione dati sul device
cudaMalloc((void**)&A_device,size);
//inizializzazione dati sull'host
initializeArray(A_host, N);
//copia dei dati dall'host al device
cudaMemcpy(A_device, A_host, size, cudaMemcpyHostToDevice);
//copia dei risultati dal device all'host
cudaMemcpy(copy, A_device, size, cudaMemcpyDeviceToHost);
printf("array sull'host\n");
stampaArray(A_host,N);
printf("array ricopiato dal device\n");
stampaArray(copy,N);
//test di correttezza
equalArray(copy, A_host,N);
//disallocazione host
free(A_host);
free(copy);
//disallocazione device
cudaFree(A_device);
exit(0);
}
void initializeArray(int *array, int n)
{
int i;
for(i=0;i<n;i++)
array[i] = i;
}
void stampaArray(int* array, int n)
{
int i;
for(i=0;i<n;i++)
printf("%d ", array[i]);
printf("\n");
}
void equalArray(int* a, int*b, int n)
{
int i=0;
while(a[i]==b[i])
i++;
if(i<n)
printf("I risultati dell'host e del device sono diversi\n");
else
printf("I risultati dell'host e del device coincidono\n");
}

