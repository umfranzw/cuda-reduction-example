#include "kernels.h"

/* This kernel is identical to the last one.
 */
__global__ void reduce(float *input, float *output, unsigned int n)
{
    // Determine this thread's various ids
    unsigned int block_size = blockDim.x;
    unsigned int thread_id = threadIdx.x;
    unsigned int block_id = blockIdx.x;

    // Calculate the index that this block's chunk of values starts at.
    // (Same as last time)
    unsigned int block_start = block_id * block_size * 2 + thread_id;
    for (unsigned int stride = block_size; stride > 0; stride /= 2)
    {
        if (thread_id < stride && // On first iteration, this will be true for all threads.
                                  // On subsequent iterations, it will ensure that we
                                  // always use the threads in the lower half of the
                                  // block (the ones with the lowest ids). This guarentees
                                  // that the remaining values will always be
                                  // contiguous in memory
            block_start + stride < n) // If we're the last block, we may be running more threads
                                // than we need - this condition makes sure they don't
                                // interfere.
        {
            input[block_start] += input[block_start + stride];
        }
        // Sync threads to prevent anyone from reading on the next iteration before everybody's
        // done writing on this one
        __syncthreads();
    }

    // Thread 0 writes this block's partial result to the output buffer.
    if (!thread_id)
    {
        output[block_id] = input[block_start];
    }
}
