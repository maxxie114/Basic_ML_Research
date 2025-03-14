#!/usr/bin/env python3
import pycuda.driver as cuda
import pycuda.autoinit  # Automatically initialize CUDA driver and context
from pycuda.compiler import SourceModule
import numpy as np

def main():
    # Print CUDA device information.
    device_count = cuda.Device.count()
    print("CUDA Device Count:", device_count)
    for i in range(device_count):
        dev = cuda.Device(i)
        print("Device {}: {}".format(i, dev.name()))
        print("  Compute Capability:", dev.compute_capability())
        print("  Total Memory: {} MB".format(dev.total_memory() // (1024 * 1024)))
    
    # Test simple memory allocation (allocate 100 bytes).
    try:
        mem = cuda.mem_alloc(100)
        print("Successfully allocated 100 bytes on GPU.")
    except cuda.Error as e:
        print("Memory allocation failed:", e)
    
    # Define a simple CUDA kernel to add two arrays.
    mod = SourceModule("""
        __global__ void add_arrays(float *a, float *b, float *c, int n) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if(idx < n)
                c[idx] = a[idx] + b[idx];
        }
    """)
    
    # Prepare small test arrays.
    n = 10
    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)
    c = np.zeros_like(a)
    
    # Allocate memory on GPU and copy host arrays.
    a_gpu = cuda.mem_alloc(a.nbytes)
    b_gpu = cuda.mem_alloc(b.nbytes)
    c_gpu = cuda.mem_alloc(c.nbytes)
    cuda.memcpy_htod(a_gpu, a)
    cuda.memcpy_htod(b_gpu, b)
    
    # Launch the kernel.
    add_arrays = mod.get_function("add_arrays")
    block_size = 256
    grid_size = (n + block_size - 1) // block_size
    add_arrays(a_gpu, b_gpu, c_gpu, np.int32(n), block=(block_size, 1, 1), grid=(grid_size, 1))
    
    # Copy the result back from GPU to CPU.
    cuda.memcpy_dtoh(c, c_gpu)
    
    # Print the input arrays and the result.
    print("Array a:", a)
    print("Array b:", b)
    print("Result array c (a + b):", c)

if __name__ == '__main__':
    main()
