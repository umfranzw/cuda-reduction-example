#include "kernels.h"

/* This kernel is identical to the last one.
 */
__global__ void reduce(float *input, float *output, unsigned int n)
{
    // Determine this thread's various ids
    unsigned int block_size = blockDim.x;
    unsigned int local_id = threadIdx.x;
    unsigned int block_id = blockIdx.x;

    // Calculate the index that this block's chunk of values starts at.
    // As last time, each thread adds 2 values, so each block adds a total of
    // block_size * 2 values.
    // Note: unlike last time, the stride here only needs to go up to block_size.
    // This is because we are eliminating the gaps that accumulated between the
    // partial results on each iteration in the last approach.
    unsigned int index = block_id * block_size * 2 + local_id;
    for (unsigned int stride = block_size; stride > 0; stride /= 2)
    {
        if (local_id < stride && // On first iteration, this will be true for all threads.
                                 // One subsequent iterations, it will ensure that we
                                 // always use the threads in the lower half of the
                                 // block (the ones with the lowest ids). This guarentees
                                 // that the remaining values will always be
                                 // contiguous in memory
            index + stride < n) // If we're the last block, we may be running more threads
                                // than we need - this condition makes sure they don't
                                // interfere.
        {
            input[index] += input[index + stride];
        }
        // Sync threads to prevent anyone from reading on the next iteration before everybody's
        // done writing on this one
        __syncthreads();
    }

    // Thread 0 write's this block's partial result to the output buffer.
    if (!local_id)
    {
        output[block_id] = input[index];
    }
}
