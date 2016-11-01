#ifndef _PERF_H
#define _PERF_H

#include "timer.h"

/* Constants */

// used to align results table - makes for nicer formatting
#define RESULTS_TABLE_WIDTH 18

/* Structs */

typedef struct Perf
{
    Timer total_timer;  // tracks total GPU time
    Timer h2d_timer;    // tracks host to device data transfer time
    Timer d2h_timer;    // tracks device to host data transfer time
    Timer kernel_timer; // tracks kernel execution time
} Perf;

/* Prototypes */

Perf create_perf();
void destroy_perf(Perf *perf);
void print_results(unsigned int n, float final_sum, Perf *perf);

#endif
