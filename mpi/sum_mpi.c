#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv){
	int i,j;
	int mynode, totalnodes;
	int sum,startval,endval,accum;
	MPI_Status status;
	
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD, &totalnodes);
	MPI_Comm_rank(MPI_COMM_WORLD, &mynode);
	
	sum=0;
	startval=1000*(mynode/totalnodes)+1;
	endval= 1000*(mynode+1)/totalnodes;
	
	for(i=startval; i<=endval; i++){
		sum+=i;
		}
	if(mynode!=0){
		MPI_Send(&sum,1,MPI_INT,0,1,MPI_COMM_WORLD);
		}
	else{
		for(j=1;j<totalnodes;j++){
			MPI_Recv(&accum,1,MPI_INT,j,1,MPI_COMM_WORLD, &status);
			sum=sum+accum;
			}
		}
	if(mynode==0)
		printf("The sum from 1 to 1000 is %d \n",sum);
	
	MPI_Finalize();	
	}
