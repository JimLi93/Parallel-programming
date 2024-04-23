//#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <unistd.h>

#include <mpi.h>
#include <omp.h>
#include <math.h>

int main(int argc, char** argv) {
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	//printf("%llu, %llu\n", r, k);
    MPI_Init(&argc, &argv);
    int mpi_rank, mpi_ranks, omp_threads, omp_thread;
    //char hostname[HOST_NAME_MAX];

    //assert(!gethostname(hostname, HOST_NAME_MAX));
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_ranks);

    //int omp_threads, omp_thread;
	unsigned long long total_pixels = 0;
	unsigned long long global_pixels = 0;
    //unsigned long long local_pixels = 0;
    unsigned long long start = r * mpi_rank/mpi_ranks;
    unsigned long long end = r*(mpi_rank+1) / mpi_ranks;
	if(mpi_rank == mpi_ranks-1){
		end = r ;
	}
	//printf("Rank %d Start %llu End %llu\n", mpi_rank, start, end);
#pragma omp parallel 
    {
		omp_threads = omp_get_num_threads();
        omp_thread = omp_get_thread_num();
        unsigned long long local_pixels = 0;
		for (unsigned long long x = start+omp_thread; x < end; x=x+omp_threads) {
			unsigned long long y = ceil(sqrtl(r*r - x*x));
			local_pixels += y;
			//local_pixels %= k;
		}
		global_pixels += local_pixels % k;
        //printf("Rank %d OMPRank %d Start %llu End %llu pixels %llu\n", mpi_rank, omp_thread, start, end, global_pixels);
	}
	//printf("Rank %d Start %llu End %llu pixels %llu\n", mpi_rank, start, end, global_pixels);
	MPI_Reduce(&global_pixels, &total_pixels, 1 , MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if(mpi_rank ==0){
        printf("%llu\n", (4 * total_pixels) % k);
    }
    

    MPI_Finalize();
    return 0;
}
