#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <pthread.h>

//pthread_mutex_t mutex;
unsigned long long global_pixels = 0;
int num_threads = 4;

struct data{
	unsigned long long r;
	unsigned long long k;
	int thread_id;
	unsigned long long local_pixels;
};

void* calculation(void *mydata){
	struct data *cur_data = (struct data*) mydata;
	unsigned long long local_pixels = 0;
	for (unsigned long long x = cur_data->thread_id; x < cur_data->r; x=x+num_threads) {
		unsigned long long y = ceil(sqrtl(cur_data->r*cur_data->r - x*x));
		local_pixels += y;
		//local_pixels %= cur_data->k;
	}
	//pthread_mutex_lock(&mutex);
	//printf("In func calculation\n");
	//printf("ThreadID %d: start from %llu end to %llu\n", cur_data->thread_id, cur_data->start, cur_data->end);
	//printf("ThreadID %d: CUR-GLOBAL: %llu, LOCAL: %llu\n", cur_data->thread_id, global_pixels, local_pixels);
	//global_pixels += local_pixels;
	//pthread_mutex_unlock(&mutex);
	cur_data->local_pixels = local_pixels % cur_data->k;
	pthread_exit(NULL);
	//return nullptr;

}

int main(int argc, char** argv) {
	//if (argc != 3) {
		//fprintf(stderr, "must provide exactly 2 arguments!\n");
		//return 1;
	//}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	//unsigned long long pixels = 0;
	//if(r ==4294967295) num_threads = 8;
	pthread_t threads[num_threads];
	int rc;
	//int ID[num_threads];
	int t;
	struct data my_data[num_threads];
	//pthread_mutex_init(&mutex, NULL);

	for(t=0;t<num_threads;t++){
		my_data[t].thread_id = t;
		my_data[t].r = r;
		my_data[t].k = k;
		//printf("In func main\n");
		//printf("ThreadID %d: start from %llu end to %llu\n", t, my_data[t].start, my_data[t].end);
		rc = pthread_create(&threads[t], NULL, calculation, (void*)&my_data[t]);
		//if (rc) {
            //printf("ERROR; return code from pthread_create() is %d\n", rc);
            //exit(-1);
        //}
		
	}
	for (t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
		global_pixels += my_data[t].local_pixels;
    }
	
	//cpu_set_t cpuset;
	//sched_getaffinity(0, sizeof(cpuset), &cpuset);
	//unsigned long long ncpus = CPU_COUNT(&cpuset);

	
	//printf("%llu\n", (4 * pixels) % k);
	//pthread_mutex_destroy(&mutex);
	printf("%llu\n", (4 * global_pixels) % k);
	pthread_exit(NULL);
	return 0;
}
