# CUDA_Optimizations
Computer Architecure - Final Project

### README ###  - Final Project - ECE 5504 - Naarayanan Rao VS	

Note: The following instructions work on a Linux based system. Not verified with Windows.

Github Link for cloning the project repo - https://github.com/vsn97/CUDA_Optimizations.git

The repository consists of three folders - common, data, kmeans. 

common - It has the make config file required to make the applications (MAKE SURE THE CONFIG FILE POINTS TO RIGHT CUDA DIRECTORY IN THE SYSTEM YOU ARE RUNNING).

data - Has all the data sets required for the input of the kmeans clustering algorithm. To generate new data sets go to the inpuGen/ folder and type the following command:

  ./datagen no.of.clusters.required no.of.features.required (Eg. ./datagen 10000 30 will give you a txt file of 10000 								points with 30 features in it)

kmeans - Each optimization is in a separate folder inside the 'kmeans' folder. The folder names and their corresponding applications are as follows:

1. async_mem_access - Asynchronous Memory Access
2. loop_unroll_threads - Loop Unrolling with Optimal Number of threads
3. loop_unroll_threads_withasync - Loop Unrolling with Optimal Number of threads with Asynchronous Memory Access
4. loop_unrolling - Loop Unrolling 
5. no_of_threads - Optimization to increase the number of threads
6. prefetch - Data prefetching using the function __prefetch__global
7. prefetch1 - Conventional Prefetching

To make/compile each optimization, redirect to that particular optimization folder and either you can directly run the application (command given below ./kmeans ...) or type the following commands:

$make clean (OPTIONAL)
$make 

To run the kmeans clustering along with the input file and the profiler, type the following command:

$ nvprof ./kmeans -m 'x' -n 'x' -r -o -i ../../data/kmeans/204800(or)819200(or)any_other_data_set.txt

Where x is any real number which decides the number of cluster centers 

Here  	

	-m = Maximum number of clusters ('x')

	-n = Minimum number of clusters ('x') 
	
	-r = Calculate the RMSE (Root Mean Square Error) (Optional)
	
	-o = Display the cluster centroids (Optional) Note: Only works when -m and -n are equal
	
	-i = Input file specifier



NOTE: In Asynchronous memory operations: cudaMemcpyAsync is implemented for only of the kernels(invert_mapping). The results are only pertaining for that particular kernel. 

Instructions to change the optimizations:

1. Change the number of threads - Open kmeans_cuda.cu -> Change the THREADS_PER_DIM to the square root of your desired threads per block. Eg. if you want 256 threads, change THREADS_PER_DIM to 16

2. Change the loop unrolling factor - Open kmeans_cuda_kernel.cu -> Change all the #pragma unroll x - where x is the factor by which you want to unroll. Eg. if you 16 factor unrolling use #pragma unroll 16

3. Prefetching - Function prototypes declared in the kmeans_cuda_kernel.cu -> Go to the __prefetch_global_l1 functions and change the argument to 2/4/16/32/64/128 depending on your desired neighbourhood value.

4. Async Memory Access - To remove asynchronous memory access - Change cudaMemCpyAsync() to cudaMemCpy and change cudaMallocHost() to cudaMalloc

5. Prefetching1 - Do not modify.

