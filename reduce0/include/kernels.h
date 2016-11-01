/* This header file is used to provide the prototypes for all device kernel functions that
  * can be called from the host. To call kernels, #include it in your main .cu source file.
  */

#ifndef _KERNELS_H
#define _KERNELS_H

/* Prototypes */

__global__ void reduce(float *input, float *output, unsigned int n);

#endif
