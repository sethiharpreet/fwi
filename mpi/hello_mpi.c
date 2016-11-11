#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv){
	
	int mynode, totalnodes;
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD, &totalnodes);
	MPI_Comm_rank(MPI_COMM_WORLD, &mynode);

	printf(" Hello world from process %d of %d \n",mynode,totalnodes);
	
	MPI_Finalize();


	}

