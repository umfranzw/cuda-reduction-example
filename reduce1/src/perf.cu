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
    perf.h2d_timer = create_timer();
    perf.d2h_timer = create_timer();
    perf.kernel_timer = create_timer();

    return perf;
}

void destroy_perf(Perf *perf)
{
    destroy_timer(&(perf->total_timer));
    destroy_timer(&(perf->h2d_timer));
    destroy_timer(&(perf->d2h_timer));
    destroy_timer(&(perf->kernel_timer));
}    

// Prints the final sum, timing info, & throughput stats
void print_results(unsigned int n, float final_sum, Perf *perf)
{
    // Note: these times are in milliseconds
    float total_time = get_time(&(perf->total_timer));
    float transfer_time = get_time(&(perf->h2d_timer)) + get_time(&(perf->d2h_timer));
    float kernel_time = get_time(&(perf->kernel_timer));
    float overhead_time = total_time - (transfer_time + kernel_time);
    float throughput = (n - 1) / (total_time / 1000); //FLOPS
    
    // Calc the perceptage of GPU time spent on data transfers, kernel execution,
    // and other overhead
    float transfer_perc = 0;
    float kernel_perc = 0;
    float overhead_perc = 0;
    
    if (total_time > 0) // try to avoid printing erroneous results if kernel or
                        // data transfers fail to execute
    {
        transfer_perc = transfer_time / total_time * 100;
        kernel_perc = kernel_time / total_time * 100;
        overhead_perc = overhead_time / total_time * 100;
    }

    printf("***********\n");
    printf("* Results *\n");
    printf("***********\n");
    
    printf("%-*s: 2^%u = %u\n", RESULTS_TABLE_WIDTH, "Data size", (unsigned int) log2f(n), n);
    printf("%-*s: %f\n", RESULTS_TABLE_WIDTH, "Final sum", final_sum);
    printf("\n");
    printf("%-*s: %f ms\n", RESULTS_TABLE_WIDTH, "Total GPU time", total_time);
    printf("%-*s: %f ms (%0.2f%%)\n", RESULTS_TABLE_WIDTH, "Data transfers", transfer_time, transfer_perc);
    printf("%-*s: %f ms (%0.2f%%)\n", RESULTS_TABLE_WIDTH, "Kernel execution", kernel_time, kernel_perc);
    printf("%-*s: %f ms (%0.2f%%)\n", RESULTS_TABLE_WIDTH, "Other overhead", overhead_time, overhead_perc);
    printf("\n");
    
    printf("%-*s: %0.2f MFLOPS \n", RESULTS_TABLE_WIDTH, "Throughput", throughput / 1000000);
}
