#include <stdio.h>
#include <argp.h>


#define N 8


enum { H, A, B };


__global__
void kernel(float* data)
{
    __constant__ float cs; // constants
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = cs[H] * idx;

    float arg = absf((cs[A] * x + cs[B]) * x * x - cs[A] * cs[B]);
    data[idx] = powf(sinf(arg), 2) / sqrtf(arg);
}


struct config
{
    enum execution_side { DEVICE, HOST } execution_side;
    unsigned k;
};


static int parse_arg(int key, const char* arg, struct argp_state* state)
{
    struct config* config = state->input;
    switch (key)
    {
        case 'c': config->execution_side = HOST; break;
        case 'g': config->execution_side = DEVICE; break;
        case 'k': config->k = atoi(arg); break;
        default: argp_failure(state, 0, EINVAL, "Invalid argument."); return EINVAL;
    }
    return 0;
}


struct argp_option options[] = {
    {"cpu", 'c', 0, 0, "Execute on CPU."},
    {"gpu", 'g', 0, 0, "Execute on GPU."},
    {0, 'k', "N", 0, "K-value."},
    { 0 }
};

struct argp argp = {options, parse_arg};


int main(int argc, char* argv[])
{
    struct config config = {};

    argp_parse(&argp, argc, argv, 0, 0, &config);

    float a[config.k];
    float local_constants[3];

    local_constants[H] = N / config.k;
    local_constants[A] = N;
    local_constants[B] = N * 2;
    cudaMemcpyToSymbol(cs, local_constants, sizeof local_constants);

    float* dev_a = NULL;
    cudaEvent_t start, stop;
    float elapsed_time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaMalloc((void**) &dev_a, sizeof a);

    cudaEventRecord(&start);
    kernel<<dim3(config.k / 512), dim3(512)>>>(dev_a);
    cudaEventRecord(&stop);
    cudaEvnetElapsedTime(&elapsed_time, start, stop);

    cudaMemcpy(a, dev_a, sizeof a, cudaMemcpyDeviceToHost);
    cudaFree(dev_a);

    printf("Elapsed time: %f\n", elapsed_time);
    if (config->flags & PRINT_FLAG) {
        printf("*** Computation results ***\n");
        for (unsigned i = 0; i < config.k; ++i) {
            printf("f(%f) = %f\n", local_constants[H] * i, a[i]);
        }
    }

    return 0;
}
