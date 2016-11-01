#include "kernels.h"

/* In this kernel, we will use shared memory. Shared memory is faster than global
 * memory. This means that if we're accessing global memory multiple times (as our
 * previous kernel did), we can gain some performance by copying the stuff we need from
 * global memory into shared memory, and then operating soley on this shared memory for
 * the rest of the kernel. However, shared memory is volatile (it disapears between kernel
 * calls), so we also need to add one extra step at the end to copy our final result from
 * shared memory back to global memory.
  *
  */
__global__ void reduce(float *input, float *output, unsigned int n)
{
    // Determine this thread's various ids
    unsigned int block_size = blockDim.x;
    unsigned int local_id = threadIdx.x;
    unsigned int block_id = blockIdx.x;

    // Determine the number of values the threads in this block will need to operate upon
    // (remember, the last block may need fewer than the others)
    unsigned int chunk_size = (block_id * block_size * 2 + block_size * 2 > n) ? n % (block_size * 2) : block_size * 2;
    // how read the line above: if we're the last block and n is not divisible by
    // (block_size * 2), then set chunk size to the number of leftover elements, otherwise
    // set chunk_size to the usual (full) number of elements (block_size * 2)

    // Declare an array in shared memory. All threads in a block have access to this
    // array. The size will be (chunk_size / 2), which is a maximum of 1024 / 2 = 512.
    // The reason we don't need chunk_size space is because we'll do an extra step
    // when we transfer our data from global to shared memory. First, we'll read half of it
    // (chunk_size / 2) from global memory and store it in the shared array. Then, we'll
    // read the other half and add it into the existing values in the shared array.
    // In other words, we're doing the first step of the reduction loop in advance. The cuts
    // down on the amount of shared memory we need to use, and saves us one memory
    // access in the long run. It also means our for loop that performs the reduction
    // can be run for one iteration than usual.
    __shared__ float shared[512]; // Note: ideally, this should be chunk_size / 2,
                                  // but CUDA forces us to use a constant...

    // Calculate the index that this block's chunk of values starts at.
    // As last time, each thread adds 2 values, so each block adds a total of
    // block_size * 2 values. Same as last time.
    unsigned int index = block_id * block_size * 2 + local_id;

    // Copy half the data from our chunk into shared memory, then add in the other half
    // (as described above).
    if (local_id < chunk_size / 2)
    {
        shared[local_id] = input[index] + input[index + chunk_size / 2];
    }
    // Since shared memory is shared by all warps running on a block (which may
    // not be synchronized), we need to sync here to make sure everybody finishes
    // the above copy before we move on.
    __syncthreads();

    // Perform the rest of the reduction, using the shared memory array.
    // Note that the starting stride is / 4 instead of /2.
    // This reflects the aforementioned fact that we already did one step of the
    // reduction when we copied the data to shared memory.
    for (unsigned int stride = chunk_size / 4; stride > 0; stride /= 2)
    {
        // we may be running more threads than we need
        if (local_id < stride)
        {
            shared[local_id] += shared[local_id + stride];
        }
        // still need to sync here as usual
        __syncthreads();
    }

    // Thread 0 write's this block's partial result to the output buffer.
    // This time that means we need to copy from *shared memory*
    // back to global memory.
    // The partial result will be in shared array index 0 (there is a *separate*
    // shared array allocated for each block).
    if (!local_id)
    {
        output[block_id] = shared[0];
    }
}
