# Test CUDA MPS for a single user

The code contains a simple Message passing (MP) [example](https://github.com/maawad/affinity-test/blob/main/affinity_test.cu#L154:L227) where one kernel is a [producer](https://github.com/maawad/affinity-test/blob/main/affinity_test.cu#L125:L136) and the other is a [consumer](https://github.com/maawad/affinity-test/blob/main/affinity_test.cu#L137:L152). The code also contains a [simple test](https://github.com/maawad/affinity-test/blob/main/affinity_test.cu#L52:L109) (copy kernel).

See [Multi-Process Service](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf) for more details.

# How to test:

1. First, set the following environment variables:
```bash
export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/home/username/mps/mps
export CUDA_MPS_LOG_DIRECTORY=/home/username/mps/log
# or any location that are accessible to the user
```
Then, start the MPS control daemon by running the command:
```bash
nvidia-cuda-mps-control -d
```

2. After starting the daemon, `nvidia-smi` should show something like this:

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

3. Clone, build, and run:
```bash
git clone https://github.com/maawad/affinity-test.git
cd affinity-test
mkdir build && cd build
cmake ..
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

4. Profile:

```bash
nsys profile -o nsys_profile ./affinity_test
```

5. Import the output file in `nsys-ui`

![](/figs/nsys-output.PNG)


6. For the copy kernel (commented out), first, recompile and profile using `nvprof`:
```bash
nvprof --analysis-metrics -f -o affinity_test.nvvp  ./affinity_test
```

Then run the `nvvp` on the host machine. Here is how to launch it on Windows:
```
nvvp -vm "C:\Program Files\Java\jre1.8.0_291\bin\java.exe"
```

Using one SM (notice that the requested one SM was rounded up to two):
![](/figs/2sms-load.PNG)

Using 50% of the SMs (notice that the SMs are assigned randomly):
![](/figs/50psms-load.PNG)

7. Shut the daemon down after finishing:
```bash
 echo quit | nvidia-cuda-mps-control
```

