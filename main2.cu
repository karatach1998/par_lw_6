#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <argp.h>


#define BLOCK_SIZE 16


void mat_out(float* mat, unsigned m, unsigned n)
{
    for (unsigned i = 0; i < m; ++i) {
        for (unsigned j = 0; j < n; ++j) {
            printf("%7.3f ", mat[i * n + j]);
        }
        printf("\n");
    }
}


void gen_mat(float* mat, unsigned m, unsigned n)
{
    for (unsigned i = 0; i < m; ++i) {
        for (unsigned j = 0; j < n; ++j) {
            mat[i * n + j] = i * n + j;
        }
    }
}


__global__
void device_kernel(float* a, int m, int n)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // if (tx < n && ty < m) {
        float tmp = a[tx * n + ty];
        __syncthreads();
        a[ty * n + tx] = a[tx * n + ty];
    // }
}


void run_using_gpu(float* mat, unsigned m, unsigned n)
{
    float* dev;

    cudaMalloc((void**)&dev, sizeof (float[m * n]));
    cudaMemcpy(dev, mat, sizeof (float[m * n]), cudaMemcpyHostToDevice);

    device_kernel<<<dim3(1, 1), dim3(BLOCK_SIZE, BLOCK_SIZE)>>>(dev, m, n);

    cudaMemcpy(mat, dev, sizeof (float[m * n]), cudaMemcpyDeviceToHost);
    cudaFree(dev);
}


void run_using_cpu(float* mat, unsigned m, unsigned n)
{
#pragma omp parallel for
    for (unsigned i = 0; i < m; ++i) {
#pragma omp parallel for
        for (unsigned j = i; j < n; ++j) {
            float tmp = mat[i * n + j];
            mat[i * n + j] = mat[j * m + i];
            mat[j * m + i] = tmp;
        }
    }
}


enum executor_type { DEVICE, HOST };

struct config
{
    enum executor_type executor;
    unsigned n;
};


static
int parse_arg(int key, char* arg, struct argp_state* state)
{
    struct config* config = (struct config*) state->input;

    switch (key)
    {
        case 'c': config->executor = HOST;   break;
        case 'g': config->executor = DEVICE; break;
        case 'n': config->n = atoi(arg);     break;
    }
    return 0;
}


struct argp_option options[] = {
    {"cpu", 'c', 0, 0, "Execute using CPU."},
    {"gpu", 'g', 0, 0, "Execute using GPU."},
    {NULL, 'n', "NUM", 0, "Specifies an matrix order."},
    { 0 }
};

struct argp argp = { options, parse_arg };


int main(int argc, char* argv[])
{
    struct config config = { };

    argp_parse(&argp, argc, argv, 0, 0, &config);

    unsigned n = config.n;
    float* m = (float*) malloc(sizeof (float[n * n]));

    gen_mat(m, n, n);

    double start = omp_get_wtime();
    switch (config.executor)
    {
        case DEVICE: run_using_gpu(m, n, n); break;
        case HOST:   run_using_cpu(m, n, n); break;
    }
    double stop = omp_get_wtime();

    mat_out(m, n, n);
    printf("** Execution time: %f\n", stop - start);

    return 0;
}
