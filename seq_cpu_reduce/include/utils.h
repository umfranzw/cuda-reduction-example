#ifndef _UTILS_H
#define _UTILS_H

/* Constants */

// If True, host array will be filled with floats in [0, 1].
// Otherwise, array will be filled with 1.0 (useful for debugging,
// since causes expected final result to be n).
#define USE_RAND_VALS 1

// used to align results table - makes for nicer formatting
#define RESULTS_TABLE_WIDTH 18

/* Prototypes */

void init_array(float *array, int len);
int parse_args(int argc, char *argv[]);
void print_array(const char *label, float *array, unsigned int len);
void print_results(int n, float result, double start_time, double stop_time);

#endif
