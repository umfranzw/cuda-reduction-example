#ifndef _REDUCE_H
#define _REDUCE_H

#include "timer.h"

/* Constants */

// If True, host array will be filled with floats in [0, 1].
// Otherwise, array will be filled with 1.0 (useful for debugging,
// since causes expected final result to be n).
#define USE_RAND_VALS 1

// If True, host will print debugging information (e.g. thread block sizes, etc.)
#define DEBUG_INFO 1

/* Prototypes */

void print_results(unsigned int n, float host_result, Timer *total_timer, Timer *copy_to_dev_timer, Timer *copy_to_host_timer, Timer *kernel_timer);

#endif
