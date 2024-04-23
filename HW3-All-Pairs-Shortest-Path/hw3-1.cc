#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <omp.h>

#include <time.h>

const int INF = ((1 << 30) - 1);
const int V = 10000;
void input(char* inFileName);
void output(char* outFileName);

void block_FW(int B);
int ceil(int a, int b);
void cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height);

int n, m;
static int Dist[V][V];
int num_of_threads;

int main(int argc, char* argv[]) {
    // Measure Time Start
    struct timespec start, end, temp;
    double time_used;
    clock_gettime(CLOCK_MONOTONIC, &start);
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    printf("%d cpus available\n", CPU_COUNT(&cpu_set));
    num_of_threads = CPU_COUNT(&cpu_set);
    omp_set_num_threads(num_of_threads);
    input(argv[1]);
    int B = 32;
    block_FW(B);
    output(argv[2]);
    // Measure Time Endx
    clock_gettime(CLOCK_MONOTONIC, &end);
    if((end.tv_nsec - start.tv_nsec) < 0){
        temp.tv_sec = end.tv_sec-start.tv_sec-1;
        temp.tv_nsec = 1000000000 +  end.tv_nsec-start.tv_nsec;
    }
    else {
        temp.tv_sec = end.tv_sec-start.tv_sec;
        temp.tv_nsec = end.tv_nsec-start.tv_nsec;
    }
    time_used = temp.tv_sec + (double) temp.tv_nsec / 1000000000.0;
    printf("%f second\n", time_used);
    return 0;
}

void input(char* infile) {
    // Measure Time Start
    struct timespec start, end, temp;
    double time_used;
    clock_gettime(CLOCK_MONOTONIC, &start);

    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    printf("n: %d\n", n);
    fread(&m, sizeof(int), 1, file);
    printf("m: %d\n", m);

    int chunksize = n/num_of_threads + 1;

    #pragma omp parallel for simd collapse(2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Dist[i][j] = INF;
        }
    }
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
            Dist[i][i] = 0;
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]][pair[1]] = pair[2];
    }
    fclose(file);

    // Measure Time Endx
    clock_gettime(CLOCK_MONOTONIC, &end);
    if((end.tv_nsec - start.tv_nsec) < 0){
        temp.tv_sec = end.tv_sec-start.tv_sec-1;
        temp.tv_nsec = 1000000000 +  end.tv_nsec-start.tv_nsec;
    }
    else {
        temp.tv_sec = end.tv_sec-start.tv_sec;
        temp.tv_nsec = end.tv_nsec-start.tv_nsec;
    }
    time_used = temp.tv_sec + (double) temp.tv_nsec / 1000000000.0;
    printf("%f second\n", time_used);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (Dist[i][j] >= INF) Dist[i][j] = INF;
        }
        fwrite(Dist[i], sizeof(int), n, outfile);
    }
    fclose(outfile);
}

int ceil(int a, int b) { return (a + b - 1) / b; }

void block_FW(int B) {
    int round = ceil(n, B);
    for (int r = 0; r < round; ++r) {
        printf("%d %d\n", r, round);
        fflush(stdout);
        /* Phase 1*/
        cal(B, r, r, r, 1, 1);

        /* Phase 2*/
        cal(B, r, r, 0, r, 1);
        cal(B, r, r, r + 1, round - r - 1, 1);
        cal(B, r, 0, r, 1, r);
        cal(B, r, r + 1, r, 1, round - r - 1);

        /* Phase 3*/
        cal(B, r, 0, 0, r, r);
        cal(B, r, 0, r + 1, round - r - 1, r);
        cal(B, r, r + 1, 0, r, round - r - 1);
        cal(B, r, r + 1, r + 1, round - r - 1, round - r - 1);
    }
}

void cal(
    int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height) {
    int block_end_x = block_start_x + block_height;
    int block_end_y = block_start_y + block_width;

    int RoundB = Round * B;
    int RoundB1 = RoundB + B;

    #pragma omp parallel for schedule(dynamic)
    for (int b_i = block_start_x; b_i < block_end_x; ++b_i) {
        
        for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {
            // To calculate B*B elements in the block (b_i, b_j)
            // For each block, it need to compute B times
            int block_internal_start_x = b_i * B;
        int block_internal_end_x = (b_i + 1) * B;
        if (block_internal_end_x > n) block_internal_end_x = n;
            int block_internal_start_y = b_j * B;
            int block_internal_end_y = (b_j + 1) * B;
            if (block_internal_end_y > n) block_internal_end_y = n;
            
            for (int k = RoundB; k < RoundB1 && k < n; ++k) {
                // To calculate original index of elements in the block (b_i, b_j)
                // For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2

                for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
                    #pragma omp simd
                    for (int j = block_internal_start_y; j < block_internal_end_y; ++j) {
                        int tmp = Dist[i][k] + Dist[k][j];
                        Dist[i][j] = tmp * (tmp < Dist[i][j]) + Dist[i][j] * (tmp >= Dist[i][j]);
                    }
                }
            }
        }
    }
}
