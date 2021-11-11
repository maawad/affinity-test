# Test CUDA MPS for a single user


See [Multi-Process Service](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf) for more details.

# How to test:

1. First start the  MPS control daemon by exporting the following:
```bash
export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/home/username/mps/mps
export CUDA_MPS_LOG_DIRECTORY=/home/username/mps/log
# or any location that are accessible to the user
```

2. After starting the daemon `nvidia-smi` should show something like this:

```bash
$ nvidia-smi
Wed Nov 10 20:50:01 2021
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 495.29.05    Driver Version: 495.29.05    CUDA Version: 11.5     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Quadro GV100        On   | 00000000:18:00.0 Off |                  Off |
| 29%   29C    P2    36W / 250W |    151MiB / 32508MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A    664535      C   nvidia-cuda-mps-server             27MiB |
+-----------------------------------------------------------------------------+
```

3. Build and run:
```bash
mkdir build
cd build
make
./affinity_test
```

The output should look like this:
```bash
Device[0]: Quadro GV100
SMs count: 80
Producer SMs = 1/80
Consumer SMs = 79/80
Creating producer_ctx
Creating consumer_ctx
Launching kernels
Ok.
```

If the GPU doesn't support MPS, you should see something like this:
```bash
$ ./affinity_test
Device[0]: NVIDIA GeForce RTX 2070 with Max-Q Design
SMs count: 36
CUDA error at /home/username/GitHub/affinity-test/affinity_test.cu 58: cudaErrorUnsupportedExecAffinity
terminate called without an active exception
Aborted
```

4. Checking the run on the profiler:

```bash
nsys profile -o nsys_profile ./affinity_test
```

5. Import the output file in `nsys-ui`

![](/figs/nsys-output.PNG)

The test code contains a simple Message passing (MP) example. Where one kernel is a producer and the other is a consumer.


The test also contains a simple test (copy kernel). Here is the SM utilization for the simple test on `nvprof`:

Using one SM (notice that the requested one SM was rounded up to two):
![](/figs/2sms-load.PNG)


Using 50% of the SMs (notice that the SMs are assigned randomly):
![](/figs/50psms-load.PNG)

To test `nvprof` profile the kernel as follows:
```bash
nvprof --analysis-metrics -f -o affinity_test.nvvp  ./affinity_test
```
Then run the `nvvp` on the host machine. Here is how to launch it on Windows:
```
nvvp -vm "C:\Program Files\Java\jre1.8.0_291\bin\java.exe"
```



