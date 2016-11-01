/* Some misc. utility functions for things like parsing command line args, array generation, and printing results.
  */

#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Fills an array with random floats in the range [0, 1],
// or the value 1.0 (depending on USE_RAND_VALS flag)
void init_array(float *array, int len)
{
    srand(time(NULL));
    for (int i = 0; i < len; i++)
    {
#if USE_RAND_VALS
        // keep the values small to avoid overflow during the summation
        array[i] = (float) rand() / RAND_MAX;
#else
        array[i] = 1.0;
#endif
    }    
}

// Reads the value of i from the command line array and returns n = 2^i
int parse_args(int argc, char *argv[])
{
    if (argc != 2)
    {
        printf("Usage: ./reduce <i, where n = 2^i>\n");
        exit(1);
    }
    
    return (int) pow(2, atoi(argv[1]));
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

void print_results(int n, float result, double start_time, double stop_time)
{
    float total_time = (stop_time - start_time); //sec
    float throughput = (n - 1) / total_time; //FLOPS

    printf("***********\n");
    printf("* Results *\n");
    printf("***********\n");
    
    printf("%-*s: 2^%d = %d\n", RESULTS_TABLE_WIDTH, "Data size", (int) log2f(n), n);
    printf("%-*s: %f\n", RESULTS_TABLE_WIDTH, "Final sum", result);
    printf("%-*s: %f ms\n", RESULTS_TABLE_WIDTH, "Total time", total_time * 1000);
    printf("\n");
    
    printf("%-*s: %0.2f MFLOPS \n", RESULTS_TABLE_WIDTH, "Throughput", throughput / 1000000);
}
