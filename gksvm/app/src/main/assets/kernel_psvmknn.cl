// parallel-svm-knn, 2016/11/28

typedef unsigned short htype;

__kernel void Dist(
	__global float *A,
	__global float *B,
	__global float *d_inDist,
	__global uchar *d_inLabel,
	const int nTests_inWork,
	const int nTrains_r,
	const int nTrains,
	const int nFeats,
	__global htype *d_inIdx,
	const int pass)
{
	int ig = get_global_id(0);

	int rid = ig/nTrains_r;	//row of distance matrix
	int cid = ig - rid*nTrains_r; //column of distance matrix

	/** for outside id, give an big value to distance table **/
	if(cid >= nTrains){
		d_inDist[rid*nTrains_r + cid] = 3000000.0;
		d_inLabel[rid*nTrains_r + cid] = 0;
		return;
	}

	int k;
	float tmpt1 = 0.0;
	float tmpt2 = 0.0;
	int start = pass*nTests_inWork*nFeats + rid*nFeats;

	for(k = 0; k < (nFeats-1); k++)
	{
		barrier(CLK_LOCAL_MEM_FENCE);//might help to achieve read one cache line
		tmpt2 = A[start + k] - B[k*nTrains + cid];
		tmpt1 += tmpt2 * tmpt2;
	}
	tmpt1 = sqrt(tmpt1);
	d_inDist[rid*nTrains_r + cid] = tmpt1;
	d_inLabel[rid*nTrains_r + cid] = (uchar)B[(nFeats-1)*nTrains + cid];
	d_inIdx[rid*nTrains_r + cid] = (htype)cid;
}


__kernel void radixsort(
	__global int* d_inDist,
	__global uchar* d_inLabel,
	__global int* d_outDist,
	__global uchar* d_outLabel,
	__global htype* d_inIdx,
	__global htype* d_outIdx,
	__local htype* loc_histo, //64*32*short
	__local htype* local_hsum, //32*short
	const int pass,
	const int radixbits,
	const int radix,
	const int nTrains_r)
{
	int it = get_local_id(0);
	int ig = get_global_id(0);
	int locSize = get_local_size(0);//64, number of split
	/* get row id and column id on threads matrix */
	int rid = ig/locSize;
	int cid = ig - rid*locSize;
	/* map thread into distance matrix, 
	* get size managed by each thread */
	int size = nTrains_r/locSize;

	int start;
	int dist_start;
	htype newpos;
	int new_dist_pos, key, shortkey, k;
	uchar keyLabel;
	htype keyIdx;
///////////////////////////////Histogram creator
	/* loc_histo, row: radix(32), column: locSize(64)  */
	for(int ir = 0; ir < radix; ir++)
		loc_histo[ir * locSize + it] = 0;
	
	dist_start = rid * nTrains_r + cid * size;
	//get start position of distance matrix 
	for(int j = 0; j < size; j++){
		k = dist_start + j;
		key = d_inDist[k];//cannot read one cache line ??
		shortkey=(( key >> (pass * radixbits)) & (radix - 1));
		loc_histo[shortkey * locSize + it]++;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
///////////////////////////////1st Scanner, activate 32 GPU threads
	if(it < radix){
		htype prev = 0;
		htype temp = 0;
		start = it * locSize;
		for(int ir = 0; ir < locSize; ir++){
			temp = loc_histo[start + ir];
			loc_histo[start + ir] = prev;
			prev = prev + temp;
		}
		local_hsum[it] = prev;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
/////////////////////////////// 2nd Scanner, activate 16 GPU threads
	if(it < (radix / 2)){
		start = 1; // replace decale with start
		for (int d = radix>>1; d > 0; d >>= 1){
		    barrier(CLK_LOCAL_MEM_FENCE);
		    if (it < d){
		      int ai = start*(2*it+1)-1;
		      int bi = start*(2*it+2)-1;
		      local_hsum[bi] += local_hsum[ai];
		    }
		    start *= 2;
		}
		if (it == 0) local_hsum[radix - 1] = 0;
		for (int d = 1; d < radix; d *= 2){
			start >>= 1;
			barrier(CLK_LOCAL_MEM_FENCE);
			if (it < d){
				int ai = start*(2*it+1)-1;
				int bi = start*(2*it+2)-1;
				htype t = local_hsum[ai];
				local_hsum[ai] = local_hsum[bi];
				local_hsum[bi] += t;
			}
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
//////////////////////////////Histogram Paste, activate 32 GPU threads
	if(it < radix){
		start = it * locSize;
		for(int ir = 0; ir < locSize; ir++)
			loc_histo[start + ir] += local_hsum[it];	
	}
	barrier(CLK_GLOBAL_MEM_FENCE);
	
//////////////////////////////Reorder
	dist_start = rid * nTrains_r + cid * size;
	for(int j = 0; j < size; j++){
		k = dist_start + j;
		key = d_inDist[k];
		keyLabel = d_inLabel[k];
		keyIdx = d_inIdx[k];

		shortkey = ((key >> (pass * radixbits)) & (radix-1));
		newpos = loc_histo[shortkey * locSize + it];
		
		//get position of output distance matrix 
		new_dist_pos = rid*nTrains_r + (int)newpos; 	
		d_outDist[new_dist_pos] = key;
		d_outLabel[new_dist_pos] = keyLabel;
		d_outIdx[new_dist_pos] = keyIdx;

		newpos++;
		loc_histo[shortkey * locSize + it] = newpos;
	}
	
}//End of kernel



__kernel void class_histogram(
   	__global uchar* d_inLabel,
	__global htype* d_ClassHist,
	const int kValue,
	const int nClass,
	const int kSplit,
	const int nTrains_r)
{
	int ig = get_global_id(0);
	int gr = get_group_id(0);
	int locSize = get_local_size(0);//32
	int rid = ig/kSplit;
	int cid = ig - rid*kSplit;

	htype tmpClassHist[9] = {0,0,0,0,0,0,0,0,0};

	int size = kValue/kSplit;
	int start = rid*nTrains_r + cid*size;
	int key;
	for(int ir=0; ir<size; ir++){
		key = d_inLabel[start + ir]-1;
		tmpClassHist[key]++;
	}

	start = rid*nClass*kSplit + cid*nClass;
	for(int ir=0; ir<nClass; ir++){
		key = start + ir;
		d_ClassHist[key] = tmpClassHist[ir];
	}
	barrier(CLK_GLOBAL_MEM_FENCE);
}


__kernel void myKernelComputation(
	__global float *A,
	const int ib,
	__global float *C,
	const int nFeats,
	const int l,
	const float gmaVal)
{
	int ig = get_global_id(0);
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
	C[ig] = 0 - gmaVal * (square_a + square_b - 2 * sum);
}

