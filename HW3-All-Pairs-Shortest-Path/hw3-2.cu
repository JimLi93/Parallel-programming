#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>
#include <omp.h>

#define BLOCKFACTOR 64
#define HALF_FACTOR 32
const int INF = ((1 << 30) - 1);
int *Dist;
int *Dev_Dist;
int old_n;
int n, m;


void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    printf("Vertex Num: %d\n", n);
    printf("Edge   Num: %d\n", m);
    /* Record the original n (matrix size) */
    old_n = n;

    /* Padding n to muliple of (BLOCKFACTOR) */
    int n_r = n % BLOCKFACTOR;
    if(n_r != 0) n = n - n_r + BLOCKFACTOR;

    /* Memory allocation */
    cudaMallocHost((void**)&Dist, sizeof(int)*n*n);

    /* Initialize the matrix */
    #pragma omp parallel for simd collapse(2)
    for (int i = 0; i < n; ++i) {
        //int IN = i * n;
        for (int j = 0; j < n; ++j) {
            Dist[i * n +j] = INF;
        }
    }

    #pragma omp simd
    for (int i = 0; i < n; ++i) {
            Dist[i * n + i] = 0;
    }

    /* Read the input */
    int * buffer = (int*) malloc(m*3*sizeof(int));
    fread(buffer, sizeof(int), m*3, file);
    #pragma omp simd
    for (int i = 0; i < m; ++i) {
        //fread(pair, sizeof(int), 3, file);
        Dist[buffer[i*3] * n + buffer[i*3+1]] = buffer[i*3+2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE *outfile = fopen(outFileName, "w");
    for (int i = 0; i < old_n; ++i) {
        fwrite(&Dist[i * n], sizeof(int), old_n, outfile);
    }
    fclose(outfile);
}

int ceil(int a, int b) { return (a + b - 1) / b; }


/*  phase 1 of algorithm (GPU kernel) */
__global__ void phase1(int *global_Dist, int round, int matrixSize) {
    __shared__ int share_Dist[BLOCKFACTOR][BLOCKFACTOR];
    int j = threadIdx.x;
    int i = threadIdx.y;
    int RB = round * BLOCKFACTOR;

    int global_idx1 = (i + RB              ) * matrixSize + (j + RB              );
    int global_idx2 = (i + RB              ) * matrixSize + (j + RB + HALF_FACTOR);
    int global_idx3 = (i + RB + HALF_FACTOR) * matrixSize + (j + RB              );
    int global_idx4 = (i + RB + HALF_FACTOR) * matrixSize + (j + RB + HALF_FACTOR);

    share_Dist[i              ][j              ] = global_Dist[global_idx1];
    share_Dist[i              ][j + HALF_FACTOR] = global_Dist[global_idx2];
    share_Dist[i + HALF_FACTOR][j              ] = global_Dist[global_idx3];
    share_Dist[i + HALF_FACTOR][j + HALF_FACTOR] = global_Dist[global_idx4];

    #pragma unroll
    for (int k = 0; k < BLOCKFACTOR; k++) {
        __syncthreads();
        share_Dist[i              ][j              ] = min(share_Dist[i              ][k] + share_Dist[k][j              ], share_Dist[i              ][j              ] );
        share_Dist[i              ][j + HALF_FACTOR] = min(share_Dist[i              ][k] + share_Dist[k][j + HALF_FACTOR], share_Dist[i              ][j + HALF_FACTOR] );
        share_Dist[i + HALF_FACTOR][j              ] = min(share_Dist[i + HALF_FACTOR][k] + share_Dist[k][j              ], share_Dist[i + HALF_FACTOR][j              ] );
        share_Dist[i + HALF_FACTOR][j + HALF_FACTOR] = min(share_Dist[i + HALF_FACTOR][k] + share_Dist[k][j + HALF_FACTOR], share_Dist[i + HALF_FACTOR][j + HALF_FACTOR] ); 
    }

    global_Dist[global_idx1] = share_Dist[i              ][j              ];
    global_Dist[global_idx2] = share_Dist[i              ][j + HALF_FACTOR];
    global_Dist[global_idx3] = share_Dist[i + HALF_FACTOR][j              ];
    global_Dist[global_idx4] = share_Dist[i + HALF_FACTOR][j + HALF_FACTOR];
}

/*  phase 2 of algorithm (GPU kernel) */
__global__ void phase2(int *global_Dist, int round, int matrixSize){
    __shared__ int share_Dist[BLOCKFACTOR][BLOCKFACTOR * 2];
    int local_round = round;
    int i = threadIdx.y;
    int j = threadIdx.x;
    int blockId = blockIdx.y;
    int row_col = blockIdx.x;

    int RB = local_round * BLOCKFACTOR;

    int j1 = j;
    int j2 = j;
    int global_idx1, global_idx2, global_idx3, global_idx4;

    if(row_col == 0){
        global_idx1 = (RB + i              ) * matrixSize + ((blockId) * BLOCKFACTOR + j              );
        global_idx2 = (RB + i              ) * matrixSize + ((blockId) * BLOCKFACTOR + j + HALF_FACTOR);
        global_idx3 = (RB + i + HALF_FACTOR) * matrixSize + ((blockId) * BLOCKFACTOR + j              );
        global_idx4 = (RB + i + HALF_FACTOR) * matrixSize + ((blockId) * BLOCKFACTOR + j + HALF_FACTOR);
        j2 += BLOCKFACTOR;
    }
    else {
        global_idx1 = ((blockId) * BLOCKFACTOR + i              ) * matrixSize + (RB + j              );
        global_idx2 = ((blockId) * BLOCKFACTOR + i              ) * matrixSize + (RB + j + HALF_FACTOR);
        global_idx3 = ((blockId) * BLOCKFACTOR + i + HALF_FACTOR) * matrixSize + (RB + j              );
        global_idx4 = ((blockId) * BLOCKFACTOR + i + HALF_FACTOR) * matrixSize + (RB + j + HALF_FACTOR);
        j1 += BLOCKFACTOR;
    }
    if (blockId == local_round) 
        return;
    int global_pivot_idx1 = (RB + i              ) * matrixSize + RB + j              ;
    int global_pivot_idx2 = (RB + i              ) * matrixSize + RB + j + HALF_FACTOR;
    int global_pivot_idx3 = (RB + i + HALF_FACTOR) * matrixSize + RB + j              ;
    int global_pivot_idx4 = (RB + i + HALF_FACTOR) * matrixSize + RB + j + HALF_FACTOR;

    share_Dist[i              ][j1              ] = global_Dist[global_idx1];
    share_Dist[i              ][j1 + HALF_FACTOR] = global_Dist[global_idx2];
    share_Dist[i + HALF_FACTOR][j1              ] = global_Dist[global_idx3];
    share_Dist[i + HALF_FACTOR][j1 + HALF_FACTOR] = global_Dist[global_idx4];

    share_Dist[i              ][j2              ] = global_Dist[global_pivot_idx1];
    share_Dist[i              ][j2 + HALF_FACTOR] = global_Dist[global_pivot_idx2];
    share_Dist[i + HALF_FACTOR][j2              ] = global_Dist[global_pivot_idx3];
    share_Dist[i + HALF_FACTOR][j2 + HALF_FACTOR] = global_Dist[global_pivot_idx4];
  
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < BLOCKFACTOR; k++) {
        //__syncthreads();
        share_Dist[i              ][j1              ] = min(share_Dist[i              ][j1              ], share_Dist[i              ][k + BLOCKFACTOR] + share_Dist[k][j              ]);
        share_Dist[i              ][j1 + HALF_FACTOR] = min(share_Dist[i              ][j1 + HALF_FACTOR], share_Dist[i              ][k + BLOCKFACTOR] + share_Dist[k][j+HALF_FACTOR]);
        share_Dist[i + HALF_FACTOR][j1              ] = min(share_Dist[i + HALF_FACTOR][j1              ], share_Dist[i + HALF_FACTOR][k + BLOCKFACTOR] + share_Dist[k][j              ]);
        share_Dist[i + HALF_FACTOR][j1 + HALF_FACTOR] = min(share_Dist[i + HALF_FACTOR][j1 + HALF_FACTOR], share_Dist[i + HALF_FACTOR][k + BLOCKFACTOR] + share_Dist[k][j+HALF_FACTOR]);

    }
global_Dist[global_idx1] = share_Dist[i              ][j1              ];
global_Dist[global_idx2] = share_Dist[i              ][j1 + HALF_FACTOR];
global_Dist[global_idx3] = share_Dist[i + HALF_FACTOR][j1              ];
global_Dist[global_idx4] = share_Dist[i + HALF_FACTOR][j1 + HALF_FACTOR];
}


/*  phase 3 of algorithm (GPU kernel) */
__global__ void phase3(int *global_Dist, int round, int matrixSize){
    __shared__ int share_Dist[BLOCKFACTOR][BLOCKFACTOR * 2];
    int local_round = round;
    int j = threadIdx.x;
    int i = threadIdx.y;
    int block_j = blockIdx.x;
    int block_i = blockIdx.y;
    int global_pivot_i = (block_i) * BLOCKFACTOR + i;
    int global_pivot_j = (block_j) * BLOCKFACTOR + j;

    int RB = local_round * BLOCKFACTOR;
    if(block_i == round || block_j == round){
        return ;
    }

    int global_idx1 = (global_pivot_i              ) * matrixSize + global_pivot_j              ;
    int global_idx2 = (global_pivot_i              ) * matrixSize + global_pivot_j + HALF_FACTOR;
    int global_idx3 = (global_pivot_i + HALF_FACTOR) * matrixSize + global_pivot_j              ;
    int global_idx4 = (global_pivot_i + HALF_FACTOR) * matrixSize + global_pivot_j + HALF_FACTOR;
    
    int global_pivot_row_idx1 = (global_pivot_i              ) * matrixSize + RB + j              ;
    int global_pivot_row_idx2 = (global_pivot_i              ) * matrixSize + RB + j + HALF_FACTOR;
    int global_pivot_row_idx3 = (global_pivot_i + HALF_FACTOR) * matrixSize + RB + j              ;
    int global_pivot_row_idx4 = (global_pivot_i + HALF_FACTOR) * matrixSize + RB + j + HALF_FACTOR;

    int global_pivot_col_idx1 = (RB + i              ) * matrixSize + global_pivot_j              ;
    int global_pivot_col_idx2 = (RB + i              ) * matrixSize + global_pivot_j + HALF_FACTOR;
    int global_pivot_col_idx3 = (RB + i + HALF_FACTOR) * matrixSize + global_pivot_j              ;
    int global_pivot_col_idx4 = (RB + i + HALF_FACTOR) * matrixSize + global_pivot_j + HALF_FACTOR;

    int share_tmp1 = global_Dist[global_idx1];
    int share_tmp2 = global_Dist[global_idx2];
    int share_tmp3 = global_Dist[global_idx3];
    int share_tmp4 = global_Dist[global_idx4];

    share_Dist[i              ][j              ] = global_Dist[global_pivot_row_idx1];
    share_Dist[i              ][j + HALF_FACTOR] = global_Dist[global_pivot_row_idx2];
    share_Dist[i + HALF_FACTOR][j              ] = global_Dist[global_pivot_row_idx3];
    share_Dist[i + HALF_FACTOR][j + HALF_FACTOR] = global_Dist[global_pivot_row_idx4];

    share_Dist[i              ][j + BLOCKFACTOR              ] = global_Dist[global_pivot_col_idx1];
    share_Dist[i              ][j + BLOCKFACTOR + HALF_FACTOR] = global_Dist[global_pivot_col_idx2];
    share_Dist[i + HALF_FACTOR][j + BLOCKFACTOR              ] = global_Dist[global_pivot_col_idx3];
    share_Dist[i + HALF_FACTOR][j + BLOCKFACTOR + HALF_FACTOR] = global_Dist[global_pivot_col_idx4];

    __syncthreads();

    #pragma unroll
    for(int k=0;k<BLOCKFACTOR;k++){
        //__syncthreads();
        share_tmp1 = min(share_Dist[i              ][k] + share_Dist[k][j + BLOCKFACTOR              ], share_tmp1); 
        share_tmp2 = min(share_Dist[i              ][k] + share_Dist[k][j + BLOCKFACTOR + HALF_FACTOR], share_tmp2); 
        share_tmp3 = min(share_Dist[i + HALF_FACTOR][k] + share_Dist[k][j + BLOCKFACTOR              ], share_tmp3); 
        share_tmp4 = min(share_Dist[i + HALF_FACTOR][k] + share_Dist[k][j + BLOCKFACTOR + HALF_FACTOR], share_tmp4);     
    }

    global_Dist[global_idx1] = share_tmp1;
    global_Dist[global_idx2] = share_tmp2;
    global_Dist[global_idx3] = share_tmp3;
    global_Dist[global_idx4] = share_tmp4;
}

void block_FW(){
    int round = ceil(n, BLOCKFACTOR);
    cudaMalloc((void**)&Dev_Dist, sizeof(int) *n * n);
    cudaMemcpy(Dev_Dist, Dist, sizeof(int) * n * n, cudaMemcpyHostToDevice);
    dim3 block(HALF_FACTOR,HALF_FACTOR);
    dim3 grid1(1);
    dim3 grid2(2, round);
    dim3 grid3(round, round);
    for (int r = 0; r < round; ++r) {  
        phase1 <<< grid1, block >>>(Dev_Dist, r, n);
        phase2 <<< grid2, block >>>(Dev_Dist, r, n);
        phase3 <<< grid3, block >>>(Dev_Dist, r, n);        
    }
    cudaMemcpy(Dist, Dev_Dist, sizeof(int) * n * n, cudaMemcpyDeviceToHost);
}


int main(int argc, char* argv[]) {
    input(argv[1]);
    block_FW();
    output(argv[2]);
    cudaFreeHost(Dist);
    cudaFree(Dev_Dist);
    return 0;
}