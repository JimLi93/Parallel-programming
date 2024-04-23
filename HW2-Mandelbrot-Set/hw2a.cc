//success with new type of packing.
//time

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <pthread.h>
#include <math.h>
#include <emmintrin.h>
#include <sys/time.h>

#define CHECK_PERIOD 20

int iters, width, height, num_threads, *image;
double left, right, lower, upper;
int cur_row = 0;
int cur_col = 0;
//bool valid;
int divide_part, divide_size, threshold_width;

struct ThreadData{
    bool valid;
    int id;
};
pthread_mutex_t mutex;

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    #pragma omp parallel for schedule(dynamic, 10)
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

void *mandelbrot(void *argv){
    struct timeval start_time;
    gettimeofday(&start_time, NULL);

    ThreadData *arg = (ThreadData*) argv;
    bool valid = arg->valid;
    int id = arg->id;
    int work_on_row;
    int start_col;
    int end_col;
    __m128d v_two = _mm_set_pd1(2);
    __m128d v_four = _mm_set_pd1(4);
    while(valid){

        // Get partition
        pthread_mutex_lock(&mutex);
        if(cur_row == height){
            valid = false;
        }
        else if(cur_col == threshold_width){
            work_on_row = cur_row;
            start_col = cur_col;
            end_col = width;
            cur_row = cur_row + 1;
            cur_col = 0;
        }
        else {
            work_on_row = cur_row;
            start_col = cur_col;
            end_col = cur_col + divide_size;
            cur_col = end_col;
        }
        pthread_mutex_unlock(&mutex);

        // Start calculate mandelbrot set of partition
        if(valid){

        double y0 = work_on_row * ((upper - lower) / height) + lower;
        __m128d v_y0 = _mm_load_pd1(&y0);
        for (int i = start_col; i < end_col; i=i+2) {
            if(i+1<width){
            double x0[2] = {i * ((right - left) / width) + left, (i+1) * ((right - left) / width) + left};
            __m128d v_x0 = _mm_load_pd(x0);
            __m128d v_x = _mm_setzero_pd();
            __m128d v_y = _mm_setzero_pd();
            __m128d v_x2 = _mm_setzero_pd();
            __m128d v_y2 = _mm_setzero_pd();
            //__m128d v_length_squared = _mm_setzero_pd();
            int repeats[2] = {0,0};
            int lock[2] = {0,0};
            while (!lock[0] || !lock[1]) {
                __m128d v_tmp = _mm_add_pd(v_x2,v_y2);
                if(!lock[0]){
                    if(repeats[0] < iters && _mm_comilt_sd(v_tmp, v_four)){
                        repeats[0]++;
                    }
                    else {
                        lock[0] = 1;
                    }
                }
                if(!lock[1]){
                    if(repeats[1] < iters && _mm_comilt_sd(_mm_shuffle_pd(v_tmp,v_tmp,1), v_four)){
                        repeats[1]++;
                    }
                    else {
                        lock[1] = 1;
                    }
                }
                v_y = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(v_x,v_y), v_two), v_y0);
                v_x = _mm_add_pd(_mm_sub_pd(v_x2,v_y2), v_x0);
                v_x2 = _mm_mul_pd(v_x,v_x);
                v_y2 = _mm_mul_pd(v_y,v_y);
            }
            image[work_on_row * width+i] = repeats[0];
            image[work_on_row * width+i+1] = repeats[1];

            }
            
            else {
            double x0 = i * ((right - left) / width) + left;

            int repeats = 0;
            double x = 0;
            double y = 0;
            double x2 = 0;
            double y2 = 0;
            //double length_squared = 0;
            while (repeats < iters && x2+y2 < 4) {
                y = 2 * x * y + y0;
                x = x2 - y2 + x0;
                x2 = x * x;
                y2 = y * y;
                ++repeats;
            }
            image[work_on_row * width + i] = repeats;
            }
        }

        }

    }

    // Get the current time after executing the thread
    struct timeval end_time;
    gettimeofday(&end_time, NULL);
    // Calculate the elapsed time
    long elapsed_time = (end_time.tv_sec - start_time.tv_sec) * 1000000 +
                        (end_time.tv_usec - start_time.tv_usec);

    // Print the elapsed time for the thread
    printf("Thread %d took %ld microseconds\n", id, elapsed_time);
    pthread_exit(NULL);

       
}

int main(int argc, char** argv) {
    struct timeval start_time;
    gettimeofday(&start_time, NULL);
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    printf("%d cpus available\n", CPU_COUNT(&cpu_set));


    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);

    /* allocate memory for image */
    image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    num_threads = CPU_COUNT(&cpu_set);
    int threadID[num_threads];
    pthread_t thread[num_threads];
    pthread_mutex_init(&mutex, NULL);


    ThreadData args[num_threads];
    divide_part = num_threads;
    divide_size = floor(width / divide_part);
    if(divide_size%2 == 1) divide_size = divide_size++;
    threshold_width = divide_size * (divide_part - 1) ;

    int rc;
    //valid = true;
    for(int i=0;i<num_threads;i++){
        args[i].valid = true;
        args[i].id = i;
        threadID[i] = i;
        //rc = pthread_create(&thread[i], NULL, mandelbrot, (void*)&(threadID[i]));
        pthread_create(&thread[i], NULL, mandelbrot, (void*)&(args[i]));
        //if(rc) {
        //    printf("ERROR; return code from pthread_create() is %d\n", rc);
        //    exit(-1);
        //}
    }

    for(int i=0;i<num_threads;i++){
        pthread_join(thread[i], NULL);
    }

    pthread_mutex_destroy(&mutex);


    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);
    // Get the current time after executing the thread
    struct timeval end_time;
    gettimeofday(&end_time, NULL);
    // Calculate the elapsed time
    long elapsed_time = (end_time.tv_sec - start_time.tv_sec) * 1000000 +
                        (end_time.tv_usec - start_time.tv_usec);

    // Print the elapsed time for the thread
    printf("Program took %ld microseconds\n", elapsed_time);
}
