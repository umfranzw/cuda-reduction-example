/* Some simple convenience functions to make timing CUDA events a little less verbose.
  */

#include "timer.h"
#include "utils.h"

Timer create_timer()
{
    Timer timer;
    cudaEventCreate(&(timer.start));
    cudaEventCreate(&(timer.stop));

    return timer;
}

void destroy_timer(Timer *timer)
{
    cudaEventDestroy(timer->start);
    cudaEventDestroy(timer->stop);
}

void start_timer(Timer *timer)
{
    cudaEventRecord(timer->start);
}

void stop_timer(Timer *timer)
{
    cudaEventRecord(timer->stop);
}

// Returns the elapsed time between the given events, in milliseconds
float get_time(Timer *timer)
{
    cudaError_t status;
    float millisec = 0;
    status = cudaEventElapsedTime(&millisec, timer->start, timer->stop);
    check_error(status, "Error calculating time between CUDA events.");
    
    return millisec;
}
