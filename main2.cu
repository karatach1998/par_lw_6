#include <stdio.h>
#include <stdlib.h>
#include <argp.h>


#define BLOCK_SIZE 16


void gen_mat(float* mat, unsigned m, unsigned n)
{
    for (unsigned i = 0; i < m; ++i) {
        for (unsigned j = 0; j < n; ++j) {
            mat[i * n + j] = i * n + j;
        }
    }
}


__global__
void device_kernel(float* a, float* b, int m, int n)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    float tmp = a[ty * n + tx];
    __syncthreads();
    a[ty * n + tx] = b[tx * n + ty];
    b[tx * n + ty] = tmp;
}


void run_using_gpu(float* mat, unsigned m, unsigned n)
{
    device_kernel<<<dim3(), dim(BLOCK_SIZE, BLOCK_SIZE)>>>(mat, mat, m, n);
}


void run_using_cpu(float* mat, unsigned m, unsigned n)
{

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
    struct config config = { 0 };

    argp_parse(&argp, argc, argv, 0, 0, &config);

    unsigned n = config.n;
    float* m = (float*) malloc(sizeof float[n * n]);

    gen_mat(m, n, n);

    switch (config.executor)
    {
        case DEVICE: run_using_gpu(m, n, n); break;
        case HOST:   run_using_cpu(m, n, n); break;
    }

    return 0;
}
