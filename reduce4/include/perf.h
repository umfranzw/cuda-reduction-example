#ifndef _PERF_H
#define _PERF_H

#include "timer.h"
#include "reduce.h"

/* Constants */

// used to align results table - makes for nicer formatting
#define RESULTS_TABLE_WIDTH 18

/* Structs */

// Note: now that we're overlapping data transfers and kernel executions, it doesn't
// really make sense to measure kernel execution time and data transfer time like we
// have in previous versions. So this struct has been modified to omit those timers.
typedef struct Perf
{
    Timer total_timer;  // tracks total GPU time
} Perf;

/* Prototypes */

Perf create_perf();
void destroy_perf(Perf *perf);
void print_results(unsigned int n, float final_sum, Perf *perf);

#endif
