#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

int main(int argc, char** argv) {
	//if (argc != 3) {
	//	fprintf(stderr, "must provide exactly 2 arguments!\n");
	//	return 1;
	//}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	int omp_threads, omp_thread;
	unsigned long long global_pixels = 0;
#pragma omp parallel
    {
        omp_threads = omp_get_num_threads();
        omp_thread = omp_get_thread_num();
		unsigned long long local_pixels = 0;
		for (unsigned long long x = omp_thread; x < r; x=x+omp_threads) {
			unsigned long long y = ceil(sqrtl(r*r - x*x));
			local_pixels += y;
			//local_pixels %= k;
		}
		global_pixels += local_pixels % k;
    }
	printf("%llu\n", (4 * global_pixels) % k);
}
