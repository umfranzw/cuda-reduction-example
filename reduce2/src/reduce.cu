/* Comp 4510 - CUDA Sum Reduction Example
 *
 * This code performs a sum reduction on an array of size n = 2^i, where i is
 * passed in as a command line arg.
 * It outputs the resulting sum along with some performance statistics.
 *
 * Changes in this version:
 * - uses pinned memory to accelerate host to device transfer (reduce.cu)
 */

#include <stdio.h>
#include <stdlib.h>
#include "reduce.h"
#include "utils.h"
#include "perf.h"
#include "kernels.h"

int main(int argc, char *argv[])
{
    // Grab n from the command line args
    const unsigned int n = parse_args(argc, argv);
    
    // Query device to find max num threads per block
    const int block_threads = get_max_block_threads();

    // Prepare timer objects to track performance
    Perf perf = create_perf();
    
    // All CUDA API calls return a status, which we must check
    cudaError_t status;

    // Host buffers
    float *host_array;
    float final_sum;

    // Device buffers
    float *dev_input;
    float *dev_output;
    float *dev_temp; // temp pointer used to flip buffers between kernel
                     // launches (see loop below)

    // Size info for kernel launches
    unsigned int threads_needed;
    unsigned int blocks;
    unsigned int remaining; // number of elements left to add
    
    // Allocate host buffer and fill it
    // Note: Unlike last time, this time we'll use *pinned memory* for the host array.
    // Pinned memory is host memory that is locked in RAM - it can't be paged out by the OS.
    // This prevents a lot of overhead when we want to transfer it to the GPU (OS doesn't 
    // page fault so it doesn't have to waste time bringing in pages from disk as the array is read
    // during the transfer).
    // To create our host buffer in pinned memory, we just use the cudaMallocHost()
    // function instead of malloc().
    status = cudaMallocHost(&host_array, n * sizeof(float));
    check_error(status, "Error allocating host buffer.");
    init_array(host_array, n);

    // Allocate device buffers
    // Note: Input buffer needs to be of size n.
    // Output buffer is used to store partial results after each kernel launch. On first launch,
    // each block will reduce block_size * 2 values down to 1 value (except last kernel, which may
    // reduce less than block_size * 2 values down to 1 value (if  n is not a multiple of block_size * 2)).
    // On subsequent launches, we need even less output buffer space.
    // Therefore output buffer size needs to be equal to the number of blocks required for first launch.
    threads_needed = n / 2; // we'll need one thread to add every 2 elements
    blocks = threads_needed / block_threads + \ // we'll need this many blocks
        (threads_needed % block_threads > 0 ? 1 : 0); // plus one extra if threads_needed
                                                      // does not evenly divide block_threads

    status = cudaMalloc(&dev_input, n * sizeof(float));
    check_error(status, "Error allocating device buffer.");
    status = cudaMalloc(&dev_output, blocks * sizeof(float));
    check_error(status, "Error allocating device buffer.");

    // Start the program timer
    start_timer(&(perf.total_timer));
    
    // Transfer the input array from host to device
    start_timer(&(perf.h2d_timer));
    status = cudaMemcpy(dev_input, host_array, n * sizeof(float), cudaMemcpyHostToDevice);
    stop_timer(&(perf.h2d_timer));
    check_error(status, "Error on CPU->GPU cudaMemcpy for host_array.");

    // Launch kernel
    // Note: We call the kernel multiple times - each call reduces the size of the array by 2.
    start_timer(&(perf.kernel_timer));
    remaining = n; // tracks number of elements left to add
    while (remaining > 1) // continue until we have a single value left (the final sum)
    {
#if DEBUG_INFO
        printf("Launching kernels:\n");
        printf("remaining: %u\n", remaining);
        printf("blocks: %u\n", blocks);
        printf("threads_needed: %u\n", threads_needed);
        printf("\n");
#endif

        // call the kernel
        reduce<<<blocks, block_threads>>>(dev_input, dev_output, remaining);
        // re-compute our size information for the next iteration
        remaining = blocks; // After the previous kernel call, each block has reduced its chunk down to a single partial sum
        threads_needed = remaining / 2; // each thread added 2 elements
        blocks = threads_needed / block_threads + (threads_needed % block_threads ? 1 : 0); // again, might need one extra block is threads_needed is not evenly
        // divisible by block_threads

        // if we will need to do another iteration, flip (swap) the device input and output buffers;
        // i.e. the output buffer from the last call becomes input buffer for the next call, and the input buffer from last call is re-used to store output for the next call.
        // Note: no data is transferred back to the host here, this is just a pointer operation
        if (remaining > 1)
        {
            dev_temp = dev_input;
            dev_input = dev_output;
            dev_output = dev_temp;
        }
    }
    stop_timer(&(perf.kernel_timer));
    // Note: the kernel launches in the loop above are asychronous, so this may not necessarily catch kernel errors...
    // If they're not caught here, they'll be caught in the check_error() call after the next blocking operation (the GPU -> CPU data transfer below).
    check_error(cudaGetLastError(), "Error launching kernel.");

    // Transfer the element in position 0 of the dev_output buffer back to the host. This is the final sum.
    start_timer(&(perf.d2h_timer));
    status = cudaMemcpy(&final_sum, dev_output, sizeof(float), cudaMemcpyDeviceToHost);
    stop_timer(&(perf.d2h_timer));
    check_error(status, "Error on GPU->CPU cudaMemcpy for final_sum.");

    // Record the final clock time
    stop_timer(&(perf.total_timer));
    // Since GPU operates asynchronously, wait until the final time has been recorded before continuing on to print the results below
    // (this synchronizes the device and the host).
    cudaEventSynchronize(perf.total_timer.stop); 

    // Display the results & performance statistics
    print_results(n, final_sum, &perf);

    // Clean up memory (both on host *AND device*!)
    status = cudaFreeHost(host_array); // Note: must use this function instead of free() to free a *pinned* buffer
    check_error(status, "Error freeing host buffer.");
    destroy_perf(&perf);
    status = cudaFree(dev_input);
    check_error(status, "Error calling cudaFree on device buffer.");
    status = cudaFree(dev_output);
    check_error(status, "Error calling cudaFree on device buffer.");
    
    return EXIT_SUCCESS;
}
