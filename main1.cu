#include <stdio.h>
#include <math.h>
#include <argp.h>


#define N 8


enum { H, A, B };


__constant__ float cs[3]; // constants


__device__
void f(float x)
{
    float arg = fabsf((cs[A] * x + cs[B]) * x * x - cs[A] * cs[B]);
    return powf(sinf(arg), 3) / sqrtf(arg);
}


__global__
void kernel(float* data)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = cs[H] * idx;

    data[idx] = f(x);
}


void run_using_gpu(float* a, unsigned k)
{
    float* dev;

    cs[H] = N / k;
    cs[A] = N;
    cs[B] = N * 2;

    float k_expanded = (k + 511) / 512 * 512;
    cudaMalloc((void**)&dev, sizeof (float[k_expanded]));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    kernel<<<(k + 511) / 512, 512>>>(dev);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(a, dev, sizeof (float[k]), cudaMemcpyDeviceToHost);
    cudaFree(dev);
}


__host__
float f(float x)
{
    float arg = fabs((a * x + b) * x * x - a * b);
    return powf(sinf(arg), 3) / sqrtf(arg);
}


void run_using_cpu(float* a, unsigned k)
{
    const float h = N / k;
    const float a = N;
    const float b = N * 2;

#pragma omp parallel for
    for (unsigned i = 0; i < k; ++i) {
        a[i] = f(h * x);
    }
}


enum execution_side
{
    DEVICE, HOST
};

enum config_flags
{
    PRINT_FLAG
};

struct config
{
    enum execution_side executor;
    unsigned k;
};


static int parse_arg(int key, char* arg, struct argp_state* state)
{
    struct config* config = (struct config*) state->input;

    switch (key)
    {
        case 'c': config->executor = HOST;      break;
        case 'g': config->executor = DEVICE;    break;
        case 'k': config->k = atoi(arg);        break;
        case ARGP_END_KEY:
                  if (config.k == 0) {
                      argp_error(state, "K must be specified.")
                  }
    }
    return 0;
}


struct argp_option options[] = {
    {"cpu", 'c', 0, 0, "Execute on CPU."},
    {"gpu", 'g', 0, 0, "Execute on GPU."},
    {NULL,  'k', "num", 0, "K-value."},
    { 0 }
};

struct argp argp = {options, parse_arg};


int main(int argc, char* argv[])
{
    struct config config = {};

    argp_parse(&argp, argc, argv, 0, 0, &config);

    float* a = (float*) malloc(sizeof (float[config.k]));

    switch (config.executor)
    {
        case DEVICE: run_using_gpu(a, config.k);
        case HOST:   run_using_cpu(a, config.k);
    }

    return 0;
}
