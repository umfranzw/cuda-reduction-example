/* Comp 4510 - CUDA Sum Reduction Example
 *
 * This code performs a sum reduction on an array of size n = 2^i, where i is
 * passed in as a command line arg.
 * It outputs the resulting sum along with some performance statistics.
 *
 * Changes in this version:
 * - uses shared memory in the kernel to speed up memory accesses (see kernel.cu)
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
    // Create an array of device input and output buffers - one pair
    // for each stream.
    float *dev_inputs[STREAMS];
    float *dev_outputs[STREAMS];
    float *dev_temp; // temp pointer used to flip buffers between kernel
                     // launches (see loop below)

    // Size info for kernel launches
    unsigned int threads_needed; // per stream
    unsigned int blocks; // per stream
    unsigned int remaining; // per stream

    // Allocate host buffer in pinned memory and fill it
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
        status = cudaMemcpyAsync(dev_inputs[i], host_array + n / STREAMS, n / STREAMS * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
        check_error(status, "Error on CPU->GPU cudaMemcpy for array." );
    }

    // Launch kernels
    remaining = n / STREAMS; // number of elements remaining *in each stream*
    while (remaining > 1)
    {
#if DEBUG_INFO
        printf("Launching kernels:\n");
        printf("remaining: %u\n", remaining);
        printf("blocks: %u\n", blocks);
        printf("block_threads: %u\n", block_threads);
        printf("threads_needed: %u\n", threads_needed);
        printf("\n");
#endif

        // Call the kernel in each stream.
        for (int i = 0; i < STREAMS; i++)
        {
            reduce<<<blocks, block_threads, 0, streams[i]>>>(dev_inputs[i], dev_outputs[i], remaining);
        }

        // re-compute our size information for the next iteration
        remaining = blocks;
        threads_needed = remaining / 2;
        blocks = threads_needed / block_threads + (threads_needed % block_threads ? 1 : 0);

        // If we will need to do another iteration, flip (swap) the device input and
        // output buffers *in each stream*
        if (remaining > 1)
        {
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

    // For each stream, transfer the element in position 0 of it's output array
    // back to the host. We'll add these values together on the CPU side further down.
    for (int i = 0; i < STREAMS; i++)
    {
        // We'll do this asynchronously using cudaMemcpyAsync too - though it doesn't
        // really save *that much* time since we're only transferring a single float...
        status = cudaMemcpyAsync(&(partial_results[i]), dev_outputs[i], sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
        check_error(status, "Error on GPU->CPU cudaMemcpy for partial_result.");
    }

    // Record the final clock time
    stop_timer(&(perf.total_timer));
    // Wait until the final time has been recorded before continuing on to print the results below
    // (this synchronizes the device and the host).
    cudaEventSynchronize(perf.total_timer.stop);

    // Add up the partial result from each stream on the host.
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
    for (int i = 0; i < STREAMS; i++)
    {
        status = cudaStreamDestroy(streams[i]);
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
