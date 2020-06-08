/* Comp 4510 - CUDA Sum Reduction Example
 *
 * This code performs a sum reduction on an array of size n = 2^i, where i is
 * passed in as a command line arg.
 * It outputs the resulting sum along with some performance statistics.
 *
 * Changes in this version:
 * - uses multiple streams to overlap data transfers with kernel execution (see below)
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
    float partial_results[STREAMS]; // this will hold the partial sum from each stream

    // Device buffers
    // This time we need an array of device input and output buffers - one pair
    // for each stream.
    float *dev_inputs[STREAMS];
    float *dev_outputs[STREAMS];
    float *dev_temp; // temp pointer used to flip buffers between kernel
                     // launches (see loop below)

    // Size info for kernel launches
    unsigned int threads_needed; // Note: this is now *per stream*
    unsigned int blocks; // Note: this is now *per stream*
    unsigned int remaining; // Note: this is now *per stream*

    // Allocate host buffer and fill it
    // Still using pinned memory like last time.
    status = cudaMallocHost(&host_array, n * sizeof(float));
    check_error(status, "Error allocating host buffer.");
    init_array(host_array, n);

    // Allocate device buffers
    threads_needed = n / 2 / STREAMS; // per stream
    blocks = threads_needed / block_threads + \
        (threads_needed % block_threads > 0 ? 1 : 0); // per stream
    
    for (int i = 0; i < STREAMS; i++)
    {
        // The kernel running in each stream will compute the sum of n / STREAMS elements
        status = cudaMalloc(&(dev_inputs[i]), n / STREAMS * sizeof(float));
        check_error(status, "Error allocating device buffer.");
        status = cudaMalloc(&(dev_outputs[i]), blocks * sizeof(float));
        check_error(status, "Error allocating device buffer.");
    }

    // Start the program timer
    start_timer(&(perf.total_timer));
    
    // Create our streams
    
    // You can think of a stream as a queue that stores GPU commands like "run this
    // kernel", "do this data transfer", etc.
    // When a CUDA program contains multiple streams, the GPU scheduler
    // chooses up to one data command *and* one kernel execution command from the
    // heads of each of the queues, and runs them simultaneously
    // (this assumes all the kernel's dependencies have been satisfied - i.e. it's not
    // waiting on any buffers to be transfered). The scheduler takes care of making sure
    // dependencies between events in the queues are satisfied, so we don't have to
    // worry about it (much...it's still good to make sure we're inserting commands into
    // the queues in a reasonably intelligent order).
    // Each stream is represented by a cudaStream_t struct - we'll store these in an array.
    cudaStream_t streams[STREAMS];
    for (int i = 0; i < STREAMS; i++)
    {
        status = cudaStreamCreate(&(streams[i]));
        check_error(status, "Error creating CUDA stream.");
    }

    // Transfer the input array from host to device
    // We need to transfer one chunk of size n / STREAMS for each stream.
    for (int i = 0; i < STREAMS; i++)
    {
        // We can use the cudaMemCpyAsync() function to perform the transfer
        // asynchronously (host doesn't block). Rather than doing the transfer right
        // away, cudaMemCpyAsync() just deposits a data transfer command
        // into the current stream (again, think of a stream like a queue that contains
        // commands for the GPU to carry out). It will be run as soon as the GPU's DMA
        // controller is free. This means it *can* overlap with kernel execution (again,
        // provided all the buffers for the kernel in question have already been
        // transfered), which is our goal. The idea:
        // 1. Transfer chunk 1 to GPU
        // 2. Run kernel on chunk 1, while simultaneously transfering chunk 2 to GPU.
        // 3. Run kernel on chunk 2, while simultaneously transfering chunk 3 to GPU.
        // etc. for all chunks.
        status = cudaMemcpyAsync(dev_inputs[i], host_array + n / STREAMS, n / STREAMS * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
        check_error(status, "Error on CPU->GPU cudaMemcpy for array." );
    }

    // Launch kernels
    // Kernels are always launched asynchronously, so calling them in this loop
    // does not cause the host to block. Instead, each call deposits a command to
    // execute the kernel into the current stream. Unlike last time, we need to tell
    // CUDA wich stream to launch each kernel in - see note in loop below for how.
    remaining = n / STREAMS; // number of elements remaining *in each stream*
    while (remaining > 1)
    {
#if DEBUG_INFO
        printf("Launching kernels:\n");
        printf("remaining: %u\n", remaining);
        printf("blocks: %u\n", blocks);
        printf("threads_needed: %u\n", threads_needed);
        printf("\n");
#endif

        // Call the kernel in each stream.
        for (int i = 0; i < STREAMS; i++)
        {
            // Here we specify two extra args in the <<<>>> brackets. The first
            // (the 0) has to do with "dynamic shared memory", which we don't need
            // to worry about, while the second (streams[i]) is the stream we want
            // to launch the kernel in. Note that our GPU can launch multiple kernels
            // at once, provided all dependencies are satistifed.
            reduce<<<blocks, block_threads, 0, streams[i]>>>(dev_inputs[i], dev_outputs[i], remaining);
        }

        // re-compute our size information for the next iteration
        remaining = blocks;
        threads_needed = remaining / 2;
        blocks = threads_needed / block_threads + (threads_needed % block_threads ? 1 : 0);

        // If we will need to do another iteration, flip (swap) the device input and
        // output buffers *in each stream*.
        if (remaining > 1)
        {
            // Note: This does not require the host to block, as it's just a pointer operation
            // (no data is transfered, and no immediate action is required on the part
            // of the device). It simply changes the pointers we'll pass to the kernel
            // calls we'll make on the next iteration.
            for (int i = 0; i < STREAMS; i++)
            {
                dev_temp = dev_inputs[i];
                dev_inputs[i] = dev_outputs[i];
                dev_outputs[i] = dev_temp;
            }
        }
    }
    // Note: the kernel launches in the loop above are asychronous, so this may not necessarily catch kernel errors...
    // If they're not caught here, they'll be caught in the check_error() call after the next blocking operation 
    // (the cudaEventSynchronize() call below).
    check_error( cudaGetLastError(), "Error in kernel." );

    // For each stream, transfer the element in position 0 of its output array
    // back to the host. We'll add these values together on the CPU to compute
    // the final sum in a loop further down.
    for (int i = 0; i < STREAMS; i++)
    {
        // We'll do this asynchronously using cudaMemcpyAsync() too - though it doesn't
        // really save *that much* time since we're only transferring a single float...
        status = cudaMemcpyAsync(&(partial_results[i]), dev_outputs[i], sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
        check_error(status, "Error on GPU->CPU cudaMemcpy for partial_result.");
    }

    // Record the final clock time
    stop_timer(&(perf.total_timer));
    // Since GPU operates asynchronously, wait until the final time has been recorded before continuing on to print the results below
    // (this synchronizes the device and the host).
    cudaEventSynchronize(perf.total_timer.stop);

    // Add up the partial results from each stream on the host.
    // Note: the fact that these few final adds are done on the CPU is factored into
    // the throughput calculation - see the print_results() function in perf.cu for more info.
    final_sum = 0;
    for (int i = 0; i < STREAMS; i++)
    {
        final_sum += partial_results[i];
    }

    // Display the results & performance statistics
    print_results(n, final_sum, &perf);

    // Clean up memory (both on host *AND device*!)
    // This time that means cleaning up multiple input/output buffers,
    // since we created one pair for each stream.
    for (int i = 0; i < STREAMS; i++)
    {
        status = cudaStreamDestroy(streams[i]); // Note: have to destroy the stream structs too...
        check_error(status, "Error destroying stream.");
        status = cudaFree(dev_inputs[i]);
        check_error(status, "Error calling cudaFree on device buffer.");
        status = cudaFree(dev_outputs[i]);
        check_error(status, "Error calling cudaFree on device buffer.");
    }
    destroy_perf(&perf);
    status = cudaFreeHost(host_array);
    check_error(status, "Error freeing host buffer.");

    return EXIT_SUCCESS;
}
