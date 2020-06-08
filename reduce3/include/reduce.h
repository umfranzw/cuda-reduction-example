#ifndef _REDUCE_H
#define _REDUCE_H

#include "timer.h"

/* Constants */

// Number of streams to use. The host array will be broken into this many chunks
// and each chunk will be transferred to the GPU asynchronously. As soon as one
// chunk has been transferred, we can start running a kernel on it. This kernel can
// run *while* the next chunk is being transferred.
#define STREAMS 16

// If True, host array will be filled with floats in [0, 1].
// Otherwise, array will be filled with 1.0 (useful for debugging,
// since causes expected final result to be n).
#define USE_RAND_VALS 1

// If True, host will print debugging information (e.g. thread block sizes, etc.)
#define DEBUG_INFO 0

#endif
