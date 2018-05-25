#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <argp.h>


#define N 8


double g_elapsedTime;


enum { H, A, B };


__constant__ float cs[3]; // constants


__device__
float f_device(float x)
{
    float arg = fabsf((cs[A] * x + cs[B]) * x * x - cs[A] * cs[B]);
    return powf(sinf(arg), 3) / sqrtf(arg);
}


__global__
void kernel(float* data)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = cs[H] * idx + 1.0f;

    data[idx] = f_device(x);
    // data[idx] = 0.5;
}


void run_using_gpu(float* a, unsigned k)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // --- Execution start ---
    float* dev = NULL;
    float local_cs[3];

    local_cs[H] = (float) N / k;
    local_cs[A] = (float) N;
    local_cs[B] = (float) N * 2;

    cudaMemcpyToSymbol(cs, local_cs, sizeof local_cs);

    unsigned k_expanded = (k + 511) / 512 * 512;
    cudaMalloc((void**)&dev, sizeof (float[k_expanded]));

    kernel<<<dim3(k_expanded / 512), dim3(512)>>>(dev);

    cudaMemcpy(a, dev, sizeof (float[k]), cudaMemcpyDeviceToHost);
    cudaFree(dev);
    // --- Execution stop ---

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    g_elapsedTime = milliseconds / 1000;
}


float f_host(float x)
{
    const float a = (float) N;
    const float b = (float) N * 2;

    float arg = fabs((a * x + b) * x * x - a * b);
    return powf(sinf(arg), 3) / sqrtf(arg);
}


void run_using_cpu(float* a, unsigned k)
{
    double start = omp_get_wtime();

    // --- Execution start ---
    const float h = (float) N / k;
    unsigned i;

#pragma omp parallel for private(i)
    for (i = 0; i < k; ++i) {
        a[i] = f_host(h * i + 1.0f);
    }
    // --- Execution stop ---

    double stop = omp_get_wtime();
    g_elapsedTime = stop - start;
}


enum execution_side
{
    DEVICE, HOST
};

enum config_flags
{
    PRINT_FLAG = 0x01,
};

struct config
{
    enum execution_side executor;
    unsigned flags;
    unsigned k;
};


static int parse_arg(int key, char* arg, struct argp_state* state)
{
    struct config* config = (struct config*) state->input;

    switch (key)
    {
        case 'c': config->executor = HOST;      break;
        case 'g': config->executor = DEVICE;    break;
        case 'p': config->flags |= PRINT_FLAG;  break;
        case 'k': config->k = atoi(arg);        break;
        case ARGP_KEY_END:
                  if (config->k == 0) {
                      argp_error(state, "K must be specified.");
                  }
    }
    return 0;
}


struct argp_option options[] = {
    {"cpu", 'c', 0, 0, "Execute on CPU."},
    {"gpu", 'g', 0, 0, "Execute on GPU."},
    {NULL,  'p', 0, 0, "Print result."},
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
        case DEVICE: run_using_gpu(a, config.k); break;
        case HOST:   run_using_cpu(a, config.k); break;
    }

    if (config.flags & PRINT_FLAG) {
        unsigned i;

        printf("** Execution result **\n");
        for (i = 0; i < config.k; ++i) {
            printf("f(%f) = %f\n", ((double)N / config.k) * i, (float)a[i]);
        }
    }
    printf("** Elapsed time: %f\n", g_elapsedTime);

    free(a);
    return 0;
}
