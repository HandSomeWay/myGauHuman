#include <stdio.h>

__global__ void hello_from_GPU()
{
    printf("hello from GPU \n");
}

int main(void){
    hello_from_GPU<<<1, 4>>>();
    cudaDeviceSynchronize();

    return 0;
}