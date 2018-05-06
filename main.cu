#include <stdio.h>


#define N (1024 * 1024)


__global__
void kernel(float* data)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = 2.0f * 3.1415926f * (float) idx / (float) N;

    data[idx] = sinf(sqrtf(x));
}


int main(int argc, char* argv[])
{
    float a[N];
    float * dev = NULL;

    cudaMalloc((void**) &dev, sizeof (float[N]));

    kernel<<<dim3(N/512), 1), dim3(512, 1)>>>(dev);

    cudaMemcpy(a, dev, sizeof (float[N]), cudaMemcpyDeviceToHost);
    cudaFree(dev);

    for (int idx = 0; idx < N; ++idx) printf("a[%d] = %.5f\n", idx, a[idx]);
    return 0;
}
