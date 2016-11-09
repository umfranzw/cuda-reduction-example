#include "kernels.h"

/* This kernel improves on the last one by using a smarter "linear indexing" approach.
 * In this approach, all threads access *consecutive* indices on each iteration of the loop.
 * This is better because it allows the hardware to take advantage of *global memory coalescing.*
 *
 * This time, each thread will add an element in the first half of the array with another one in
 * the second half. This means that collectively, the threads will access
 * consective array indices when they read from the right half, and when they (read and) write to the left half.
 * Like the previous approach, the number of threads required still halves on each iteration.
 * However, unlike the previous approach, the distance between the elements being added
 * does *not* double on each iteration, because the results are written back to the first half
 * of the array in a consecutive fashion. This can significantly improve the performance of
 * the kernel because coalescing cuts down on the number of memory requests that need
 * to be made (several large requests rather than many individual ones).
 */
__global__ void reduce(float *input, float *output, unsigned int n)
{
    // Determine this thread's various ids
    unsigned int block_size = blockDim.x;
    unsigned int thread_id = threadIdx.x;
    unsigned int block_id = blockIdx.x;

    // Calculate the index that this block's chunk of values starts at.
    // As last time, each thread adds 2 values, so each block adds a total of
    // block_size * 2 values.
    // Note: unlike last time, the stride here only needs to go up to block_size.
    // This is because we are eliminating the gaps that accumulated between the
    // partial results in the last approach.
    unsigned int block_start = block_id * block_size * 2 + thread_id;
    for (unsigned int stride = block_size; stride > 0; stride /= 2)
    {
        if (thread_id < stride && // On first iteration, this will be true for all threads.
                                  // On subsequent iterations, it will ensure that we
                                  // always use the threads in the lower half of the
                                  // block (the ones with the lowest ids). This guarentees
                                  // that the remaining values will always be
                                  // contiguous in memory (unlike the last approach,
                                  // which left gaps between them)
            block_start + stride < n) // If we're the last block, we may be running more threads
                                      // than we need - this condition makes sure they don't
                                      // interfere.
        {
            input[block_start] += input[block_start + stride];
        }
        // As last time, we need to sync.
        __syncthreads();
    }

    // As last time, thread 0 writes this block's partial result to the output buffer.
    if (!thread_id)
    {
        output[block_id] = input[block_start];
    }
}
