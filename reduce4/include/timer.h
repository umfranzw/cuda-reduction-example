#ifndef _TIMER_H
#define _TIMER_H

/* Structs */

typedef struct Timer
{
    cudaEvent_t start;
    cudaEvent_t stop;
} Timer;

/* Prototypes */

Timer create_timer();
void destroy_timer(Timer *timer);
void start_timer(Timer *timer);
void stop_timer(Timer *timer);
float get_time(Timer *timer);

#endif
