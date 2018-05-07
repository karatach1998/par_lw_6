#include <stdio.h>
#include <stdlib.h>
#include <argp.h>


// __global__
// void kernel(float* data)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
// }


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
    float m[n * n];

    printf("%i %u\n", config.executor, config.n);

    return 0;
}
