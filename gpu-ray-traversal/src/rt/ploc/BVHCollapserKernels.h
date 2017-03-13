/**
* \file		BVHCollapserKernels.h
* \author	Daniel Meister
* \date		2017/02/23
* \brief	BVHCollapser kernels header file.
*/

#ifndef _BVH_COLLAPSER_KERNELS_H_
#define _BVH_COLLAPSER_KERNELS_H_

#include "CudaBVHNode.h"

#define BVH_COLLAPSER_BLOCK_THREADS 256

namespace FW {

#ifdef __CUDACC__
extern "C" {

	__device__ int interiorPrefixScanOffset;
	__device__ int leafPrefixScanOffset;
	__device__ int prefixScanOffset;

	__global__ void computeNodeStatesAdaptive(
		const int numberOfTriangles,
		const float ci,
		const float ct,
		int * termCounters,
		float * nodeCosts,
		int * nodeParentIndices,
		int * nodeLeftIndices,
		int * nodeRightIndices,
		int * nodeSizes,
		int * nodeStates,
		Vec4f * nodeBoxesMin,
		Vec4f * nodeBoxesMax
	);

	__global__ void computeNodeStates(
		const int numberOfTriangles,
		const int maxLeafSize,
		int * termCounters,
		int * nodeParentIndices,
		int * nodeSizes,
		int * nodeStates
	);

	__global__ void computeLeafIndices(
		const int threads,
		const int numberOfTriangles,
		int * leafIndices,
		int * nodeParentIndices,
		int * nodeStates
	);

	__global__ void invalidateCollapsedNodes(
		const int threads,
		const int numberOfTriangles,
		int * termCounters,
		int * leafIndices,
		int * nodeParentIndices,
		int * nodeSizes,
		int * nodeStates
	);

	__global__ void computeNodeOffsets(
		const int threads,
		const int numberOfNodes,
		int * nodeOffsets,
		int * nodeStates
	);

	__global__ void computeTriangleOffsets(
		const int threads,
		const int numberOfNodes,
		int * nodeStates,
		int * nodeSizes,
		int * triangleOffsets
	);

	__global__ void compact(
		const int numberOfNodes,
		const int newNumberOfInteriorNodes,
		int * nodeStates,
		int * nodeOffsets,
		int * nodeParentIndices,
		int * nodeLeftIndices,
		int * nodeRightIndices,
		int * nodeSizes,
		int * triangleOffsets,
		Vec4f * nodeBoxesMin,
		Vec4f * nodeBoxesMax,
		CudaBVHNode * nodes
	);

	__global__ void reorderTriangles(
		const int threads,
		const int numberOfTriangles,
		int * triangleOffsets,
		int * inputTriangleIndices,
		int * outputTriangleIndices,
		int * leafIndices
	);

	__global__ void convert(
		const int threads,
		const int numberOfNodes,
		int * nodeParentIndices,
		int * nodeLeftIndices,
		int * nodeRightIndices,
		int * nodeSizes,
		Vec4f * nodeBoxesMin,
		Vec4f * nodeBoxesMax,
		CudaBVHNode * nodes
	);

}
#endif

};

#endif /* _BVH_COLLAPSER_KERNELS_H_ */
