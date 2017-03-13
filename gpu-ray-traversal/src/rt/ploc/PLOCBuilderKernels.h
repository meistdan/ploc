/**
* \file		PLOCBuilderKernels.h
* \author	Daniel Meister
* \date		2017/01/23
* \brief	PLOCBuilder kernels header file.
*/

#ifndef _PLOC_BUILDER_KERNELS_H_
#define _PLOC_BUILDER_KERNELS_H_

#include "CudaBVHNode.h"

#define PLOC_SCAN_BLOCK_THREADS 1024
#define PLOC_REDUCTION_BLOCK_THREADS 256
#define PLOC_GEN_BLOCK_THREADS 256
#define PLOC_BLOCK_THREADS 256

namespace FW {

#ifdef __CUDACC__
extern "C" {

	__constant__ float sceneBoxConst[6];
	__device__ float sceneBox[6];
	__device__ float cost;
	__device__ int prefixScanOffset;

	__global__ void computeSceneBox(
		const int threads,
		const int numberOfVertices
	);

	__global__ void computeMortonCodes30(
		const int threads,
		const int numberOfTriangles,
		Vec4f * nodeBoxesMin,
		Vec4f * nodeBoxesMax,
		int * triangleIndices,
		unsigned int * mortonCodes
		);

	__global__ void computeMortonCodes60(
		const int threads,
		const int numberOfTriangles,
		Vec4f * nodeBoxesMin,
		Vec4f * nodeBoxesMax,
		int * triangleIndices,
		unsigned long long * mortonCodes
		);

	__global__ void setupClusters(
		const int threads,
		const int numberOfTriangles,
		int * nodeLeftIndices,
		int * nodeRightIndices,
		int * nodeSizes,
		int * nodeIndices
		);

	__global__ void generateNeighboursCached(
		const int numberOfClusters,
		const int radius,
		float * neighbourDistances,
		int * neighbourIndices,
		Vec4f * nodeBoxesMin,
		Vec4f * nodeBoxesMax,
		int * nodeIndices
		);

	__global__ void generateNeighbours(
		const int numberOfClusters,
		const int radius,
		float * neighbourDistances,
		int * neighbourIndices,
		Vec4f * nodeBoxesMin,
		Vec4f * nodeBoxesMax,
		int * nodeIndices
		);

	__global__ void merge(
		const int numberOfClusters,
		const int nodeOffset,
		int * neighbourIndices,
		int * nodeIndices0,
		int * nodeIndices1,
		int * nodeParentIndices,
		int * nodeLeftIndices,
		int * nodeRightIndices,
		int * nodeSizes,
		Vec4f * nodeBoxesMin,
		Vec4f * nodeBoxesMax
		);

	__global__ void localPrefixScan(
		const int numberOfClusters,
		int * nodeIndices,
		int * threadOffsets,
		int * blockOffsets
		);

	__global__ void globalPrefixScan(
		const int numberOfBlocks,
		int * blockOffsets
		);

	__global__ void compact(
		const int numberOfClusters,
		int * nodeIndices0,
		int * nodeIndices1,
		int * blockOffsets,
		int * threadOffsets
		);

	__global__ void woopifyTriangles(
		const int threads,
		const int numberOfTriangles,
		int * triangleIndices,
		Vec4f * triWoopsA,
		Vec4f * triWoopsB,
		Vec4f * triWoopsC
	);

	__global__ void computeCost(
		const int threads,
		const int numberOfNodes,
		const float sceneBoxArea,
		const float ct,
		const float ci,
		CudaBVHNode * nodes
	);

}
#endif

};

#endif /* _PLOC_BUILDER_KERNELS_H_ */
