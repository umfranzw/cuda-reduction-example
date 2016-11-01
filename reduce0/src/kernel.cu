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
 * No cudaMemCpy is require to do this (cudaMemCpy is only needed for non-primitive data types).
 */
__global__ void reduce(float *input, float *output, unsigned int n)
{
    // Determine this thread's various ids
    unsigned int block_size = blockDim.x;
    unsigned int local_id = threadIdx.x;
    unsigned int block_id = blockIdx.x;

    // Calculate the index that this block's chunk of values starts at.
    // Each thread adds 2 values, so each block adds a total of block_size * 2 values.
    unsigned int index = block_id * (block_size * 2);
    
    // Perform the reduction using "interleaved indexing" (each thread adds a pair of
    // elements right next to each other).
    // The stride is the distance between the elements that each thread adds.
    // This distance doubles on each iteration.
    // Note that the number of threads required also halves on each iteration.
    unsigned int left;  // holds index of left operand
    unsigned int right; // holds index or right operand
    for (unsigned int stride = 1; stride < block_size * 2; stride *= 2)
    {
        left = index + local_id * stride * 2;
        right = left + stride;
        // we may be running more threads than we need
        if (right - index < block_size * 2 // read: "If this thread should be
                                           // active on this iteration."
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
    // for this block's chunk. The code below causes each block's thread 0 to write that
    // partial result to the output buffer at the index given by its block_id. After the code
    // below completes, the output buffer will contain exactly <number of blocks>
    // consecutive partial results.
    if (!local_id)
    {
        output[block_id] = input[index];
    }
}
