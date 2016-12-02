/* version 10-18 */


__kernel void myKernelComputation(
	__global float *A,
	/*__global float *B,*/
	const int ib,
	__global float *C,
	const int nFeats,
	const int l,
	const float gmaVal,
	const int start,
	const int len)
{
	int ig = get_global_id(0);
	if(ig < start) return;
	if(ig >= len) return;
	
	int ir;
	int key, key2;
	float tmp = 0.0;
	float sum = 0.0;
	float square_a = 0.0;
	float square_b = 0.0;

	for(ir=0; ir<nFeats; ir++)
	{
		key = ir*l + ig;
		key2 = ir*l + ib;
		square_a += A[key] * A[key];
		square_b += A[key2] * A[key2];

		tmp = A[key] * A[key2];
		sum += tmp;
	}

    //tmp = exp( 0 - gmaVal * (square_a + square_b - 2 * sum) );
	//C[ig] = tmp;
	C[ig] = 0 - gmaVal * (square_a + square_b - 2 * sum);
}

__kernel void myMatrixComputation(
	__global float *A,
	__global float *B,
	__global float *C,
	const int nFeats,
	const int l)
{
	int ig = get_global_id(0);
	int ir;
	float tmp = 0.0;
	float sum = 0.0;

	for(ir=0; ir<nFeats; ir++)
	{
		tmp = A[ir*l + ig] - B[ir];
		sum += tmp * tmp;
	}

	C[ig] = sum;
}