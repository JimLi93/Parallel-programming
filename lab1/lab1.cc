#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>

int main(int argc, char** argv)
{
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  unsigned long long r = atoll(argv[1]);
  unsigned long long k = atoll(argv[2]);
  unsigned long long pixels = 0;
  unsigned long long tmp = r*r;
  for(unsigned long long x = rank; x<r;x=x+size){
    unsigned long long y = ceil(sqrtl(tmp-x*x));
    pixels+=y;
    pixels %= k;
  }
  unsigned long long total;
  MPI_Reduce(&pixels, &total, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  if(rank==0){
    printf("%llu\n", (4*total)%k);
  }
  MPI_Finalize();
}
