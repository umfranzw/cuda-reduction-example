#include "kernels.h"

/* This is the kernel function that performs the sum reduction.
 * This one use the most basic approach (called "interleaved indexing").
 * Each thread adds a pair of elements right next to each other.
 * The stride in the code below is the distance between the elements that each thread adds.
 * This distance doubles on each iteration.
 * Note that the number of threads required also halves on each iteration.
 *
 * Notes on kernel args: the arguments passed in for the arrays must be *device buffers* (not host buffers)!
 * n is an integer that is passed in from the host when the kernel is launched.
 * No cudaMemCpy is require to do this (args on the stack are copied).
 */
__global__ void reduce(float *input, float *output, unsigned int n)
{
    // Determine this thread's various ids
    unsigned int block_size = blockDim.x;
    unsigned int thread_id = threadIdx.x;
    unsigned int block_id = blockIdx.x;

    // The size of the chunk of data this thread's block is working on.
    unsigned int chunk_size = block_size * 2;

    // Calculate the index that this block's chunk of values starts at.
    // Each thread adds 2 values, so each block adds a total of block_size * 2 values.
    unsigned int block_start = block_id * chunk_size;
    
    // Perform the reduction using "interleaved indexing" (each thread adds a pair of
    // elements right next to each other).
    // "stride" is the distance between the elements that each thread adds.
    // This distance doubles on each iteration.
    // The number of threads required halves on each iteration.
    unsigned int left;  // holds index of left operand
    unsigned int right; // holds index or right operand
    unsigned int threads = block_size; // number of active threads (on current iteration)
    for (unsigned int stride = 1; stride < chunk_size; stride *= 2, threads /= 2)
    {
        // There's a distance of stride between each pair of left and right operand indices,
        // so there's a distance of stride * 2 between consecutive left indices
        left = block_start + thread_id * (stride * 2);
        right = left + stride;

        if (thread_id < threads // read: "If this thread should be
                                // active on this iteration of the reduction."
            && right < n) // If we're the last block, we may be running more threads
                          // than we need - this condition makes sure they dont interfere.
        {
            input[left] += input[right];
        }
        // Each block may be running multiple warps. These warps may not all be in
        // sync. The call below syncs the warps in the block at the end of each iteration
        // so that the results are written to memory before the next iteration begins.
        __syncthreads();
    }

    // Once the loop is done, the partial sum for this block will be in the leftmost index
    // of this block's chunk. The code below causes each block's thread 0 to write that
    // partial result to the output buffer at position "block_id". After the code
    // below completes, the output buffer will contain exactly <number of blocks>
    // consecutive partial results.
    if (!thread_id)
    {
        output[block_id] = input[block_start];
    }
}
