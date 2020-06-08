/* Comp 4510 - Sequential CPU-Based Sum Reduction Example
 *
 * This code performs a sum reduction on an array of size n = 2^i, where i is
 * passed in as a command line arg.
 * It outputs the resulting sum along with some performance statistics.
 *
 * This is a sequential CPU implmentation to establish baseline for comparison with GPU versions.
 * We include the OpenMP library only for its timing functions (manually timing things in C is nasty...),
 * we're not parallelizing anything here.
 */

#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include <time.h>
#include <omp.h>

float reduce(float *array, int n)
{
    float total = 0;

    for (int i = 0; i < n; i++)
    {
        total += array[i];
    }

    return total;
}

int main(int argc, char *argv[])
{
    //Grab i from the command line and calculate n
    const int n = parse_args(argc, argv);

    float *array;
    double start_time, stop_time;
    float total_sum;

    // Allocate & fill an array of n floats
    array = (float *) malloc(n * sizeof(float));
    init_array(array, n);

    // Do the reduction
    start_time = omp_get_wtime();
    total_sum = reduce(array, n);
    stop_time = omp_get_wtime();

    // Display the results, along with some throughput stats
    print_results(n, total_sum, start_time, stop_time);

    // clean up
    free(array);

    return EXIT_SUCCESS;
}
