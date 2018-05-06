#include <stdio.h>


#define N 8

__global__
void kernel(float* data)
{
    enum { H, A, B };
    __constant__ float cs; // constants
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = cs[H] * idx;

    float arg = absf((cs[A] * x + cs[B]) * x * x - cs[A] * cs[B]);
    data[idx] = powf(sinf(arg), 2) / sqrtf(arg);
}


int main(int argc, char* argv[])
{

    return 0;
}
