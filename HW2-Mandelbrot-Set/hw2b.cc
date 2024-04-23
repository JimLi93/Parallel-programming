
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
#include <mpi.h>
#include <omp.h>

int iters, width, height, num_threads;
double left, right, lower, upper;

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

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    //printf("%d cpus available\n", CPU_COUNT(&cpu_set));

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

    int rank,size;
    MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
	MPI_Comm_size(MPI_COMM_WORLD, &size);

    int q = height / size; 
    int r = height % size;
    
    int row_total = 0;
    if(rank < r){
        row_total = q + 1;
    } 
    else row_total = q;

    /* allocate memory for image */
    int* local_image = (int*)malloc(width * row_total * sizeof(int));
    assert(local_image);
 
    __m128d v_two = _mm_set_pd1(2);
    __m128d v_four = _mm_set_pd1(4);
	#pragma omp parallel 
    {
        #pragma omp for schedule(dynamic) 
	for(int j=0;j<row_total;j++){
		int work_on_row = j*size+rank;
		double y0 = work_on_row * ((upper - lower) / height) + lower;
        __m128d v_y0 = _mm_load_pd1(&y0);
        for (int i = 0; i < width; i=i+2) {
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
            local_image[j * width+i] = repeats[0];
            local_image[j * width+i+1] = repeats[1];

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
            local_image[j* width + i] = repeats;
            }
        }
		
	}
	}
    int* misordered_image = (int*)malloc(width * height * sizeof(int));
    int recvcounts[size];
    int displs[size];

    for(int i=0;i<size;i++){
        if(i < r){
            recvcounts[i] = (q+1) * width;
        }
        else {
            recvcounts[i] = q * width;
        }
        if(i == 0){
            displs[i] = 0;
        }
        else {
            displs[i] = displs[i-1] + recvcounts[i-1];
        }
    }
    
    MPI_Gatherv(local_image, width * row_total, MPI_INT, misordered_image, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);


    if(rank == 0) {
        int* ordered_image = (int*)malloc(width * height * sizeof(int));
        #pragma omp parallel
        {
            #pragma omp for schedule(dynamic)
            for(int i=0;i<size;i++){
                for(int j=i, tmp=0;j<height;j=j+size, tmp++){
                    for(int k=0;k<width;k++){
                        ordered_image[j*width+k] = misordered_image[displs[i]+tmp*width+k];
                    }
                }
            }
        }
        write_png(filename, iters, width, height, ordered_image);
    }

    /* draw and cleanup */
    free(local_image);
    MPI_Finalize();

}
