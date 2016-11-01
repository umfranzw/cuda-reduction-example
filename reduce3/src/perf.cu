/* Some simple convenience functions used to collect, calculate, & print  performance statistics.
 * The perf struct (see header file) contains several timer objects that can be used to track events.
  */

#include "perf.h"
#include "timer.h"
#include <stdio.h>

Perf create_perf()
{
    Perf perf;
    perf.total_timer = create_timer();

    return perf;
}

void destroy_perf(Perf *perf)
{
    destroy_timer(&(perf->total_timer));
}    

// Prints the final sum, timing info, & throughput stats.
// Note: now that we're overlapping data transfers and kernel executions, it doesn't
// really make sense to measure kernel execution time and data transfer time like we
// have in previous versions. So this function has been modified to omit those stats.
void print_results(unsigned int n, float final_sum, Perf *perf)
{
    // Note: this time is in milliseconds
    float total_time = get_time(&(perf->total_timer));
    // Note: unlike in the previous version version, here, in the throughput calculation
    // below, we subtract off an extra (STREAMS - 1) from the numerator because the
    // CPU now does that many adds (the multiple kernel execution streams generate
    // STREAMS partial results which the CPU sums - see reduce.cu for details)
    float throughput = (n - 1 - (STREAMS - 1)) / (total_time / 1000); //FLOPS

    printf("***********\n");
    printf("* Results *\n");
    printf("***********\n");
    
    printf("%-*s: 2^%u = %u\n", RESULTS_TABLE_WIDTH, "Data size", (unsigned int) log2f(n), n);
    printf("%-*s: %f\n", RESULTS_TABLE_WIDTH, "Final sum", final_sum);
    printf("\n");
    printf("%-*s: %f ms\n", RESULTS_TABLE_WIDTH, "Total GPU time", total_time);
    printf("\n");
    
    printf("%-*s: %0.2f MFLOPS \n", RESULTS_TABLE_WIDTH, "Throughput", throughput / 1000000);
}
