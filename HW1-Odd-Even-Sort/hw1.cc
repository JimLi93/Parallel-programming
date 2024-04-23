#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <boost/sort/spreadsort/float_sort.hpp>

void merge_array(float *arr1, float *arr2, float *t, int len1, int len2, int type);

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);

    int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get array length
	int total_n = atoi(argv[1]);

    // max_rank = max process - 1
    int max_rank;
	if (total_n < size) max_rank = total_n - 1;
	else max_rank = size - 1;
    
    // calculate partial array length of the process(rank)
    // 0~max_recv_idx -> (+1) others -> (+0)
	int min_part_n = total_n / (double) size;
	int max_recv_idx = (total_n % size) - 1;
    int part_n = min_part_n;
    if(rank<=max_recv_idx) part_n = part_n + 1;

    // calculate the address start from process(rank) (To read and write file)
    int part_start;
    if(rank<=max_recv_idx) part_start = (min_part_n + 1) * rank;
    else part_start = min_part_n * rank + max_recv_idx + 1;
    
    // Allocate spaces for  temp, buff, data
    float *buff = new float[min_part_n+1];
    float *temp = new float[min_part_n+1];
	float *data = new float[part_n];

	int prev_part_n, next_part_n;

    if(rank + 1 <= max_recv_idx) next_part_n = min_part_n + 1;
    else next_part_n = min_part_n;
	
    if(rank - 1 <= max_recv_idx) prev_part_n = min_part_n + 1;
    else prev_part_n = min_part_n;

    MPI_File f1, f2;

	MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &f1);
	MPI_File_read_at(f1, sizeof(float) * part_start, data, part_n, MPI_FLOAT, MPI_STATUS_IGNORE);
	MPI_File_close(&f1);

	// sort on partial array
    if(part_n != 0){
        boost::sort::spreadsort::float_sort(data, data + part_n);
    }
	
    int total_valid = 0;
    int cur_valid = 0;
    bool isEven = (rank % 2 == 0);

	while (total_valid < max_rank + 1)
	{
        // suppose current phase is valid. will be check if unvalid
        cur_valid = 1;
        
        // For process with data 
        if(part_n != 0){

            // EVEN PHASE

		    if (isEven && rank != max_rank){
			    MPI_Sendrecv(data, part_n, MPI_FLOAT, rank + 1, 0, buff, next_part_n, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if(data[part_n-1] > buff[0]) {
                    cur_valid = 0;
			        merge_array(data, buff, temp, part_n, next_part_n, 0);
                } 
		    }
		    else if (!isEven) {
			    MPI_Sendrecv(data, part_n, MPI_FLOAT, rank - 1, 0, buff, prev_part_n, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if((buff[prev_part_n-1] > data[0])) {
                    cur_valid = 0;
			        merge_array(buff, data, temp, prev_part_n, part_n, 1);
                }
		    }
            
            //ODD PHASE

		    if (!isEven && rank != max_rank){
			    MPI_Sendrecv(data, part_n, MPI_FLOAT, rank + 1, 0, buff, next_part_n, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    			if(data[part_n-1] > buff[0]){
                    cur_valid = 0;
                    merge_array(data, buff, temp, part_n, next_part_n, 0);
                }
		    }
    		else if (isEven && rank != 0) {
	    		MPI_Sendrecv(data, part_n, MPI_FLOAT, rank - 1, 0, buff, prev_part_n, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		    	if(buff[prev_part_n-1] > data[0]) {
                    cur_valid = 0;
			        merge_array(buff, data, temp, prev_part_n, part_n, 1);
                }
		    }
		}
		MPI_Allreduce(&cur_valid, &total_valid, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	}

	MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &f2);
	MPI_File_write_at(f2, sizeof(float) * part_start, data, part_n, MPI_FLOAT, MPI_STATUS_IGNORE);
	MPI_File_close(&f2);

	delete[] data;
    delete[] buff;
	MPI_Finalize();
	return 0;
}

void merge_array(float *arr1, float *arr2, float *t, int len1, int len2, int type)
{
    if(type == 0){
	    int i = 0, j = 0, k = 0;
        while(i<len1 && j<len2 && k<len1){
            if(arr1[i] < arr2[j]) t[k++] = arr1[i++];
            else {
                t[k++] = arr2[j++];
            }
        }
        while(k < len1 && i < len1){
            t[k++] = arr1[i++];
        }
        for(int h=0;h<len1;h++){
            arr1[h] = t[h];
        }
    }
    else if(type == 1){
        int i = len1-1, j = len2-1, k = len2-1;
        while(i>=0 && j>=0 && k>=0){
            if(arr1[i] <= arr2[j]) t[k--] = arr2[j--];
            else {
                t[k--] = arr1[i--];
            }
        }
        while(k >= 0 && j >= 0){
            t[k--] = arr2[j--];
        }

        for(int h=0;h<len2;h++){
            arr2[h] = t[h];
        }
    }
    return;
}