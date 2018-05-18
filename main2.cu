#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>
#include <argp.h>


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


#define TILE_SIZE 16


__global__
void transpose_optimized(float* a, int n)
{
    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    tile[threadIdx.y][threadIdx.x] = a[y * n + x];

    __syncthreads();

    a[y * n + x] = tile[threadIdx.x][threadIdx.y];
}


void run_using_gpu(float* mat, unsigned m, unsigned n)
{
    printf("[%u %u]\n", m, n);
    float* dev;

    assert(m == n);
    assert(m % TILE_SIZE == 0 && n % TILE_SIZE == 0);

    cudaMalloc((void**)&dev, sizeof (float[m * n]));
    cudaMemcpy(dev, mat, sizeof (float[m * n]), cudaMemcpyHostToDevice);

    uint3 blocks = dim3(m / TILE_SIZE, n / TILE_SIZE);
    uint3 threads = dim3(TILE_SIZE, TILE_SIZE);
    transpose_optimized<<<blocks, threads>>>(dev, m, n);

    cudaMemcpy(mat, dev, sizeof (float[m * n]), cudaMemcpyDeviceToHost);
    cudaFree(dev);
}


void run_using_cpu(float* mat, unsigned m, unsigned n)
{
#pragma omp parallel for collapse(2)
    for (unsigned i = 0; i < m; ++i) {
        for (unsigned j = i + 1; j < n; ++j) {
            float tmp = mat[i * n + j];
            mat[i * n + j] = mat[j * m + i];
            mat[j * m + i] = tmp;
        }
    }
}


enum executor_type { DEVICE, HOST };
enum flags { PRINT_RESULT = 0x1 }

struct config
{
    enum executor_type executor;
    enum flags flags;
    unsigned n;
};


static
int parse_arg(int key, char* arg, struct argp_state* state)
{
    struct config* config = (struct config*) state->input;

    switch (key)
    {
        case 'c': config->executor = HOST;      break;
        case 'g': config->executor = DEVICE;    break;
        case 'p': config->flags = PRINT_RESULT; break;
        case 'n': config->n = atoi(arg);        break;
        case ARGP_END_KEY:
                  if (config->n == 0) {
                      argp_error(state, "Matrix dimestions must be specified.");
                      fflush(stdout);
                      return -1;
                  }
    }
    return 0;
}


struct argp_option options[] = {
    {"cpu", 'c', 0, 0, "Execute using CPU."},
    {"gpu", 'g', 0, 0, "Execute using GPU."},
    {NULL,  'p', 0, 0, "Print result matrix."},
    {NULL,  'n', "NUM", 0, "Specifies an matrix order."},
    { 0 }
};

struct argp argp = { options, parse_arg };


int main(int argc, char* argv[])
{
    struct config config = { };

    argp_parse(&argp, argc, argv, 0, 0, &config);

    unsigned n = config.n;
    float* mat = (float*) malloc(sizeof (float[n * n]));

    gen_mat(mat, n, n);

    double start = omp_get_wtime();
    switch (config.executor)
    {
        case DEVICE: run_using_gpu(mat, n, n); break;
        case HOST:   run_using_cpu(mat, n, n); break;
    }
    double stop = omp_get_wtime();

    if (config.flags & PRINT_RESULT)
        mat_out(mat, m, n);
    printf("** Execution time: %f\n", stop - start);

    return 0;
}
