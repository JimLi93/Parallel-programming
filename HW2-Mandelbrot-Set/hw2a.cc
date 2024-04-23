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

struct ThreadData{
    bool idle;
    int id;
};

int iters, width, height, num_threads, *image;
double left, right, lower, upper;
int cur_row = 0;
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
    ThreadData *arg = (ThreadData*) argv;
    bool idle = arg->idle;
    int id = arg->id;
    int work_on_row;
    bool end_thread = false;
    while(1){
        if(idle == false){
            work_on_row = id;
            //printf("Thread %d working on row %d\n", id, work_on_row);
            double y0 = work_on_row * ((upper - lower) / height) + lower;
            for (int i = 0; i < width; ++i) {
                double x0 = i * ((right - left) / width) + left;
                int repeats = 0;
                double x = 0;
                double y = 0;
                double length_squared = 0;
                while (repeats < iters && length_squared < 4) {
                    double temp = x * x - y * y + x0;
                    y = 2 * x * y + y0;
                    x = temp;
                    length_squared = x * x + y * y;
                    ++repeats;
                }
                image[work_on_row * width + i] = repeats;
            }
            idle = true;
        }
    
        pthread_mutex_lock(&mutex);
        if(cur_row >= height){
            end_thread = true;
        }
        else {
            work_on_row = cur_row;
            cur_row = cur_row + 1;
            //end_thread = false;
        }
        pthread_mutex_unlock(&mutex);

        if(end_thread == true) {
            pthread_exit(NULL);
        }
        
        else if(end_thread == false){
            //printf("Thread %d working on row %d\n", id, work_on_row);
            double y0 = work_on_row * ((upper - lower) / height) + lower;
            for (int i = 0; i < width; ++i) {
                double x0 = i * ((right - left) / width) + left;
                int repeats = 0;
                double x = 0;
                double y = 0;
                double length_squared = 0;
                while (repeats < iters && length_squared < 4) {
                    double temp = x * x - y * y + x0;
                    y = 2 * x * y + y0;
                    x = temp;
                    length_squared = x * x + y * y;
                    ++repeats;
                }
                image[work_on_row * width + i] = repeats;
            }
        }
    }

   pthread_exit(NULL);
       
}

int main(int argc, char** argv) {
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
    int rc;
    int max_threads = 0;
    cur_row = num_threads;
    
    for(int i=0;i<num_threads;i++){
        threadID[i] = i;
        if(i >= height){
            max_threads = i;
            printf("Early exit\n");
            break;
        }
        else{
            args[i].idle = false;
            args[i].id = i;
            pthread_create(&thread[i], NULL, mandelbrot, (void*)&(args[i]));
        } 
        //rc = pthread_create(&thread[i], NULL, mandelbrot, (void*)&(threadID[i]));
        //if(rc) {
        //    printf("ERROR; return code from pthread_create() is %d\n", rc);
        //    exit(-1);
        //}
    }
    printf("%d %d\n", max_threads, num_threads);
    if(max_threads == 0) max_threads = num_threads;
    printf("%d %d\n", max_threads, num_threads);

    for(int i=0;i<max_threads;i++){
        pthread_join(thread[i], NULL);
    }

    pthread_mutex_destroy(&mutex);

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);
}
