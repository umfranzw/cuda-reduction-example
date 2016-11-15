/* Some misc. utility functions for things like error checking, array generation, and device querying.
  */

#include "utils.h"
#include "reduce.h"
#include <stdio.h>

// Checks if an error occurred using the given status.
// If so, prints the given message and halts.
void check_error(cudaError_t status, const char *msg)
{
    if (status != cudaSuccess)
    {
        const char *errorStr = cudaGetErrorString(status);
        printf("%s:\n%s\nError Code: %d\n\n", msg, errorStr, status);
        exit(status); // bail out immediately before the rest of the train derails...
    }
}

// Returns the maximum number of supported threads per block on the current device.
// Note: This is a hardware-enforced limit.
int get_max_block_threads()
{
    int dev_num;
    int max_threads;
    cudaError_t status;

    // Grab the device number of the default CUDA device.
    status = cudaGetDevice(&dev_num);
    check_error(status, "Error querying device number.");

    // Query the max possible number of threads per block
    status = cudaDeviceGetAttribute(&max_threads, cudaDevAttrMaxThreadsPerBlock, dev_num);
    check_error(status, "Error querying max block threads.");

    return max_threads;
}

// Fills an array with random floats in the range [0, 1],
// or the value 1.0 (depending on USE_RAND_VALS flag)
void init_array(float *array, int len)
{
    srand(time(NULL));
    
    int i;
    for (i = 0; i < len; i++)
    {
        // keep the values small to avoid overflow during the summation
#if USE_RAND_VALS
        array[i] = (float) rand() / RAND_MAX;
#else
        array[i] = 1.0;
#endif
    }    
}

// Prints the given array to stdout (for debugging)
void print_array(const char *label, float *array, unsigned int len)
{
    printf("%s", label);
    
    int i;
    for (i = 0; i < len; i++)
    {
        printf("%f ", array[i]);
    }
    printf("\n\n");
}

// Reads the value of i from the command line array and returns n = 2^i
unsigned int parse_args(int argc, char *argv[])
{
    if (argc != 2)
    {
        printf("Usage: ./reduce <i, where n = 2^i>\n");
        exit(1);
    }
    
    return (unsigned int) pow(2, atoi(argv[1]));
}
