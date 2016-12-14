// parallel-knn, 2016/11/28C

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
	const int pass)
{
	int ig = get_global_id(0);

	int rid = ig/nTrains_r;	//距离方阵的行
	int cid = ig - rid*nTrains_r; //距离方阵的列

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
		tmpt2 = A[start + k] - B[k*nTrains + cid];
		tmpt1 += tmpt2 * tmpt2;
	}
	tmpt1 = sqrt(tmpt1);
	d_inDist[rid*nTrains_r + cid] = tmpt1;
	d_inLabel[rid*nTrains_r + cid] = (uchar)B[(nFeats-1)*nTrains + cid];
}


__kernel void histogram(
	__global int* d_inDist,
	__local htype* loc_histo,
	const int pass,
	__global htype* d_Histograms,
	const int radixbits,
	const int nTrains_r,
	const int split,
	const int radix)
{
	int it = get_local_id(0);
	int ig = get_global_id(0);
  	int locSize = get_local_size(0);
	int Width = nTrains_r;
	int rid = ig/Width;
	int cid = ig - rid*Width;
	int size = nTrains_r/split;

 	int start = rid*Width + cid*size;
 	int key, shortkey, k;

	for(int ir=0;ir<radix;ir++){
		loc_histo[ir * locSize + it] = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	for(int j= 0; j< size; j++){

  		k = start + j;
		key = d_inDist[k];

  		shortkey=(( key >> (pass * radixbits)) & (radix-1));
  		loc_histo[shortkey * locSize + it]++;
  	}
  	barrier(CLK_LOCAL_MEM_FENCE);

  	start = rid*radix*split + cid*radix;

  	for(int ir=0;ir<radix;ir++){

  		k = start + ir;
    		d_Histograms[k] = loc_histo[ir * locSize + it];
	}
	barrier(CLK_GLOBAL_MEM_FENCE);
}

__kernel void scanhistograms(
	__global htype* histo,
	__local htype* temp,
	__global htype* globsum,
	const int split)
{

	int it = get_local_id(0);
	int ig = get_global_id(0);
	int radix = get_local_size(0); //radix
	int gr = get_group_id(0);

	int start = gr*radix*split;//_RADIX*SPLIT代表一行test数据个数

	for(int ir=0; ir<split; ir++){
		temp[ir*radix + it] = histo[start + ir*radix + it];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	htype tempz = 0; //temp value of histogram
	htype tempi;
	for(int ir=0; ir<split; ir++){
		tempi = temp[ir*radix + it];
		temp[ir*radix + it] = tempz;//将前一个累计hist值赋予
		tempz = tempz + tempi;//将前一个累计hist值 加上当前hist值
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	globsum[gr*radix + it] = tempz;
	barrier(CLK_GLOBAL_MEM_FENCE);

	/**  copy to global memory of d_histgram  **/
	for(int ir=0; ir<split; ir++){
		histo[start + ir*radix + it] = temp[ir*radix + it];
	}
	barrier(CLK_GLOBAL_MEM_FENCE);
}

//gloal size: nTests_r*_RADIX/2, local size: _RADIX/2
//local memory size: _RADIX*sizeof(htype)
__kernel void scanhistograms2(
	__global htype* globsum,
	__local htype* temp,
	__global htype* gsum)
{
	int it = get_local_id(0);
	int ig = get_global_id(0);

	int decale = 1;

	int n = get_local_size(0) * 2; // 16*2 = 32
	int gr = get_group_id(0);

	// local mem: temp 64 int
	temp[2*it] = globsum[2*ig];
	temp[2*it+1] = globsum[2*ig+1];

	for (int d = n>>1; d > 0; d >>= 1){
	    barrier(CLK_LOCAL_MEM_FENCE);
	    if (it < d){
	      int ai = decale*(2*it+1)-1;
	      int bi = decale*(2*it+2)-1;
	      temp[bi] += temp[ai];
	    }
	    decale *= 2;
	}

	if (it == 0) {
   		gsum[gr]=temp[n-1];
   		temp[n - 1] = 0;
	}

	for (int d = 1; d < n; d *= 2){

		decale >>= 1;
		barrier(CLK_LOCAL_MEM_FENCE);

		if (it < d){
			int ai = decale*(2*it+1)-1;
			int bi = decale*(2*it+2)-1;

			htype t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}

	}
	barrier(CLK_LOCAL_MEM_FENCE);

	globsum[2*ig] = temp[2*it];
	globsum[2*ig+1] = temp[2*it+1];

	barrier(CLK_GLOBAL_MEM_FENCE);
}

__kernel void pastehistograms(
	__global htype* histo,
	__global htype* globsum,
	const int split)
{
	int it = get_local_id(0);
	int ig = get_global_id(0);
	int gr = get_group_id(0);
	int locSize = get_local_size(0);

	htype gsum;
	gsum = globsum[ig];

  	int start = gr*locSize*split;

	for(int ir=0; ir<split; ir++){
		histo[start + ir*locSize + it] += gsum;
	}
	barrier(CLK_GLOBAL_MEM_FENCE);
}

__kernel void reorder(
        __global int* d_inDist,
        __global uchar* d_inLabel,
        __global int* d_outDist,
        __global uchar* d_outLabel,
        __global htype* d_Histograms,
        __local htype* loc_histo,
        const int pass,
        const int split,
	const int nTrains_r,
	const int radixbits,
	const int radix)
{

	int it = get_local_id(0);
	int ig = get_global_id(0);
	int gr = get_group_id(0);
	int locSize = get_local_size(0);

	int rid = ig/split;
	int cid = ig - rid*split;

	int size = nTrains_r/split;

	int start = rid*radix*split + radix*cid;

	for(int ir=0; ir<radix; ir++){
		loc_histo[ir*locSize + it] = d_Histograms[start + ir];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	htype newpos;
	int newposition, key, shortkey, k;
	uchar keyLabel;

	start = rid*nTrains_r + cid*size;
	for(int j=0; j<size; j++){

		k = start + j;
		key = d_inDist[k];
		keyLabel = d_inLabel[k];

		shortkey = ((key >> (pass * radixbits)) & (radix-1));
		newpos = loc_histo[shortkey * locSize + it];

		newposition = rid*nTrains_r + (int)newpos;

		d_outDist[newposition] = key;
		d_outLabel[newposition] = keyLabel;

		newpos++;
		loc_histo[shortkey * locSize + it] = newpos;
	}

}


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

__kernel void bitonicMerge(
	__global float * inDist,
	__global uchar* inLabel,
	__local float * tmpDist,
	__local uchar* tmpLabel,
	const int numGC,
	const int order)
{
	int it = get_local_id(0);
	int ig = get_global_id(0);
	int gr = get_group_id(0);
	int wg = get_local_size(0); 

	//row and colum of group cluster
	int rid = ig/(wg * numGC);
	int cid = gr - rid * numGC;
	int scope = wg * order;

	//global memory offset
	int offset = rid * scope * numGC * 2 + cid * scope;
	//int offset = rid * 256 * wgn + cid * 256;
	inDist += offset;
	inLabel += offset;

	for(int ir=it; ir<scope; ir+=wg)
	{
		int op1 = 2 * scope - 1 - ir;
		int op2 = scope * (numGC * 2 - cid * 2) - 1 - ir;
		tmpDist[ir] = inDist[ir];
		tmpDist[op1] = inDist[op2];
		tmpLabel[ir] = inLabel[ir];
		tmpLabel[op1] = inLabel[op2];
		barrier(CLK_LOCAL_MEM_FENCE);

		//ComparatorLocal(aux[it], aux[op1]);
		if(tmpDist[ir] > tmpDist[op1]){
		    float tmp;
		    tmp = tmpDist[ir];
			tmpDist[ir] = tmpDist[op1];
			tmpDist[op1] = tmp;

			uchar utmp;
			utmp = tmpLabel[ir];
			tmpLabel[ir] = tmpLabel[op1];
			tmpLabel[op1] = utmp;
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		inDist[ir] = tmpDist[ir];
		inDist[op2] = tmpDist[op1];
		inLabel[ir] = tmpLabel[ir];
		inLabel[op2] = tmpLabel[op1];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}

__kernel void bitonicSort(
	__global float* inDist,
	__global uchar* inLabel,
	const int split)
{
	int it = get_local_id(0);
	int wg = get_local_size(0);
	int data_len = wg*split;//case 1: 64*4=256, case 2: 64*16=1024
	//global memory offset
	int offset = get_group_id(0) * data_len; //case 1: i*64*4, case 2: i*64*16
	inDist += offset;
	inLabel += offset;

	// Loop on sorted sequence length
	for (int length=1; length<data_len; length<<=1)
	{
		// Loop on comparison distance (between keys)
		for (int inc=length; inc>0; inc>>=1)
		{
			float tmpDist[16];
			uchar tmpLabel[16];
			// Each thread work on several(data_len/wg) data
			for(int i=it; i<data_len; i+=wg){

				bool direction = (( i & (length<<1)) != 0); // direction of sort: 0=asc, 1=desc
				int j = i ^ inc; // sibling to compare

				float iKey = inDist[i];
				float jKey = inDist[j];
				uchar iLabel = inLabel[i];
				uchar jLabel = inLabel[j];

				bool smaller = (jKey < iKey) || ( jKey == iKey && j < i ); //j is smaller than i
				bool swap = smaller ^ (j < i) ^ direction;

				tmpDist[i/wg] = (swap)?jKey:iKey;
				tmpLabel[i/wg] = (swap)?jLabel:iLabel;
			}
			barrier(CLK_LOCAL_MEM_FENCE);

			for(int i=it; i<data_len; i+=wg)
			{
				inDist[i] = tmpDist[i/wg];
				inLabel[i] = tmpLabel[i/wg];
			}
			barrier(CLK_LOCAL_MEM_FENCE);

		}
	}
}

__kernel void bitonicSelect(
	__global float* inDist,
	__global uchar* inLabel,
	const int kValue,
	const int len,
	const int numPt)
{
	int ig = get_global_id(0);
	int rid = ig/numPt;
	int cid = ig - rid*numPt;

	int offset1 = ig * len;
	int offset2 = rid * len * numPt + cid * kValue;

	for(int ir=0; ir<kValue; ++ir)
	{
		inDist[offset2 + ir] = inDist[offset1 + ir];
		inLabel[offset2 + ir] = inLabel[offset1 + ir];
	}

}


__kernel void bitonicSort2(
	__global float* inDist,
	__global uchar* inLabel,
	__local float* locDist, //256
	__local uchar* locLabel,
	const int split,
	const int nTrains_r)
{
	int it = get_local_id(0);
	int wg = get_local_size(0); // //64

	int data_len = wg * split;//64*4=256

	int offset = get_group_id(0) * nTrains_r; //i*32k
	inDist += offset;
	inLabel += offset;

	for(int ir=it; ir<data_len; ir+=wg)
	{
		locDist[ir] = inDist[ir];
		locLabel[ir] = inLabel[ir];
	}
	barrier(CLK_LOCAL_MEM_FENCE); // make sure AUX is entirely up to date

	// Loop on sorted sequence length
	for (int length=1; length<data_len; length<<=1)
	{
		// Loop on comparison distance (between keys)
		for (int inc=length; inc>0; inc>>=1)
		{
			float tmpDist[4];
			uchar tmpLabel[4];

			for(int i=it; i<data_len; i+=wg){

				bool direction = (( i & (length<<1)) != 0); // direction of sort: 0=asc, 1=desc
				int j = i ^ inc; // sibling to compare

				float iKey = locDist[i];
				float jKey = locDist[j];
				uchar iLabel = locLabel[i];
				uchar jLabel = locLabel[j];

				bool smaller = (jKey < iKey) || ( jKey == iKey && j < i ); //j is smaller than i
				bool swap = smaller ^ (j < i) ^ direction;

				//barrier(CLK_LOCAL_MEM_FENCE);

				tmpDist[i/wg] = (swap)?jKey:iKey;
				tmpLabel[i/wg] = (swap)?jLabel:iLabel;
			}
			barrier(CLK_LOCAL_MEM_FENCE);

			for(int i=it; i<data_len; i+=wg){
				locDist[i] = tmpDist[i/wg];
				locLabel[i] = tmpLabel[i/wg];
			}
			barrier(CLK_LOCAL_MEM_FENCE);

		}
	}

	for(int i=it; i<data_len; i+=wg)
	{
		inDist[i] = locDist[i];
		inLabel[i] = locLabel[i];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

}


