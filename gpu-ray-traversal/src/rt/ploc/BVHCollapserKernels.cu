/**
* \file	BVHCollapserKernels.cu
* \author	Daniel Meister
* \date	2016/03/15
* \brief	BVHCollapser kernels soruce file.
*/

#include "BVHCollapserKernels.h"
#include "CudaBVHUtil.cuh"

using namespace FW;

extern "C" __global__ void computeNodeStatesAdaptive(
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
	) {

	// Triangle index.
	const int triangleIndex = blockDim.x * blockIdx.x + threadIdx.x;

	if (triangleIndex < numberOfTriangles) {

		// Node index.
		int nodeIndex = triangleIndex + numberOfTriangles - 1;

		// Box.
		AABB box;
		box.grow(Vec3f(nodeBoxesMin[nodeIndex].x, nodeBoxesMin[nodeIndex].y, nodeBoxesMin[nodeIndex].z));
		box.grow(Vec3f(nodeBoxesMax[nodeIndex].x, nodeBoxesMax[nodeIndex].y, nodeBoxesMax[nodeIndex].z));

		// Cost.
		nodeCosts[nodeIndex] = ci * box.area();
		nodeStates[nodeIndex] = 0;

		// Actual node index.
		nodeIndex = nodeParentIndices[nodeIndex];

		// Go up to the root.
		while (atomicAdd(&termCounters[nodeIndex], 1) > 0) {

			// Box.
			AABB box;
			box.grow(Vec3f(nodeBoxesMin[nodeIndex].x, nodeBoxesMin[nodeIndex].y, nodeBoxesMin[nodeIndex].z));
			box.grow(Vec3f(nodeBoxesMax[nodeIndex].x, nodeBoxesMax[nodeIndex].y, nodeBoxesMax[nodeIndex].z));

			// Node.
			int nodeLeftIndex = nodeLeftIndices[nodeIndex];
			int nodeRightIndex = nodeRightIndices[nodeIndex];

			// Cost.
			float area = box.area();
			float cost = ct * area + nodeCosts[nodeLeftIndex] + nodeCosts[nodeRightIndex];
			float costAsLeaf = ci * area * nodeSizes[nodeIndex];

			// Leaf.
			if (costAsLeaf < cost) {
				nodeCosts[nodeIndex] = costAsLeaf;
				nodeStates[nodeIndex] = 0;
			}

			// Interior.
			else {
				nodeCosts[nodeIndex] = cost;
				nodeStates[nodeIndex] = 1;
			}

			// Root.
			if (nodeIndex == 0) break;

			// Go to the parent.
			nodeIndex = nodeParentIndices[nodeIndex];

		}

	}

}

extern "C" __global__ void computeNodeStates(
	const int numberOfTriangles,
	const int maxLeafSize,
	int * termCounters,
	int * nodeParentIndices,
	int * nodeSizes,
	int * nodeStates
	) {

	// Triangle index.
	const int triangleIndex = blockDim.x * blockIdx.x + threadIdx.x;

	if (triangleIndex < numberOfTriangles) {

		// Node index.
		int nodeIndex = triangleIndex + numberOfTriangles - 1;

		// Leaf.
		nodeStates[nodeIndex] = 0;

		// Actual node index.
		nodeIndex = nodeParentIndices[nodeIndex];

		// Go up to the root.
		while (atomicAdd(&termCounters[nodeIndex], 1) > 0) {

			// Leaf.
			if (nodeSizes[nodeIndex] <= maxLeafSize) {
				nodeStates[nodeIndex] = 0;
			}

			// Interior.
			else {
				nodeStates[nodeIndex] = 1;
			}

			// Root.
			if (nodeIndex == 0) break;

			// Go to the parent.
			nodeIndex = nodeParentIndices[nodeIndex];

		}

	}

}

extern "C" __global__ void computeLeafIndices(
	const int threads,
	const int numberOfTriangles,
	int * leafIndices,
	int * nodeParentIndices,
	int * nodeStates
	) {

	// Thread index.
	const int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

	for (int triangleIndex = threadIndex; triangleIndex < numberOfTriangles; triangleIndex += threads) {

		// Node.
		int nodeIndex = triangleIndex + numberOfTriangles - 1;

		// Find leaf index.
		int leafIndex = nodeIndex;
		int parentIndex = nodeParentIndices[leafIndex];
		int parentState = nodeStates[parentIndex];
		while (parentIndex > 0) {
			if (parentState == 0)
				leafIndex = parentIndex;
			nodeIndex = parentIndex;
			parentIndex = nodeParentIndices[nodeIndex];
			parentState = nodeStates[parentIndex];
		}

		// Write leaf index.
		leafIndices[triangleIndex] = leafIndex;

	}

}

extern "C" __global__ void invalidateCollapsedNodes(
	const int threads,
	const int numberOfTriangles,
	int * termCounters,
	int * leafIndices,
	int * nodeParentIndices,
	int * nodeSizes,
	int * nodeStates
	) {

	// Thread index.
	const int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

	for (int triangleIndex = threadIndex; triangleIndex < numberOfTriangles; triangleIndex += threads) {

		// Leaf index.
		int leafIndex = leafIndices[triangleIndex];

		// Node index.
		int nodeIndex = triangleIndex + numberOfTriangles - 1;

		// Leaf reached.
		if (nodeIndex == leafIndex) continue;

		// Invalidate node.
		nodeStates[nodeIndex] = -1;

		// Actual node index.
		nodeIndex = nodeParentIndices[nodeIndex];

		// Go up to the root.
		while (atomicAdd(&termCounters[nodeIndex], 1) > 0) {

			// Leaf reached.
			if (nodeIndex == leafIndex) break;

			// Invalidate node.
			nodeStates[nodeIndex] = -1;

			// Root.
			if (nodeIndex == 0) break;

			// Go to the parent.
			nodeIndex = nodeParentIndices[nodeIndex];

		}

	}

}

extern "C" __global__ void computeNodeOffsets(
	const int threads,
	const int numberOfNodes,
	int * nodeOffsets,
	int * nodeStates
	) {

	// Thread index.
	const int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

	// Warp thread index.
	const int warpThreadIndex = threadIdx.x & (WARP_THREADS - 1);

	for (int nodeIndex = threadIndex; nodeIndex < numberOfNodes; nodeIndex += threads) {

		// Node state.
		int nodeState = nodeStates[nodeIndex];

		// Node offset.
		int nodeOffset;

		// Prefix scan.
		unsigned int warpBallot = __ballot(nodeState > 0 && nodeIndex != 0);
		int warpCount = __popc(warpBallot);
		int warpIndex = __popc(warpBallot & ((1u << warpThreadIndex) - 1));

		// Add count of components to the global counter.
		int warpOffset;
		if (warpThreadIndex == 0)
			warpOffset = atomicAdd(&interiorPrefixScanOffset, warpCount);

		// Exchange offset between threads.
		warpOffset = __shfl(warpOffset, 0);
		if (nodeState > 0) nodeOffset = warpOffset + warpIndex;

		// Prefix scan.
		warpBallot = __ballot(nodeState == 0);
		warpCount = __popc(warpBallot);
		warpIndex = __popc(warpBallot & ((1u << warpThreadIndex) - 1));

		// Add count of components to the global counter.
		if (warpThreadIndex == 0)
			warpOffset = atomicAdd(&leafPrefixScanOffset, warpCount);

		// Exchange offset between threads.
		warpOffset = __shfl(warpOffset, 0);
		if (nodeState == 0) nodeOffset = warpOffset + warpIndex;

		// Node offset.
		if (nodeIndex == 0) nodeOffset = 0;
		nodeOffsets[nodeIndex] = nodeOffset;

	}

}

extern "C" __global__ void computeTriangleOffsets(
	const int threads,
	const int numberOfNodes,
	int * nodeStates,
	int * nodeSizes,
	int * triangleOffsets
	) {

	// Thread index.
	const int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

	// Warp thread index.
	const int warpThreadIndex = threadIdx.x & (WARP_THREADS - 1);

	// Node end.
	const int nodeEnd = divCeilLog(numberOfNodes, LOG_WARP_THREADS) << LOG_WARP_THREADS;

	for (int nodeIndex = threadIndex; nodeIndex < nodeEnd; nodeIndex += threads) {

		// Node state.
		int nodeState;

		// Leaf size.
		int leafSize = 0;

		// Valid node index.
		if (nodeIndex < numberOfNodes) {
			nodeState = nodeStates[nodeIndex];
			if (nodeState == 0) leafSize = nodeSizes[nodeIndex];
		}

		// Leaf size prefix scan.
		int warpSum = warpScan(warpThreadIndex, leafSize);

		// Add count to the global counter.
		int warpOffset;
		if (warpThreadIndex == 31)
			warpOffset = atomicAdd(&prefixScanOffset, warpSum);
		warpSum -= leafSize;

		// Exchange offset between threads.
		warpOffset = __shfl(warpOffset, 31);

		// Leaf.
		if (nodeIndex < numberOfNodes && nodeState == 0) {
			triangleOffsets[nodeIndex] = warpOffset + warpSum;
		}

	}

}

extern "C" __global__ void compact(
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
	) {

	// Node index.
	const int nodeIndex = blockDim.x * blockIdx.x + threadIdx.x;

	if (nodeIndex < numberOfNodes) {

		// Node state.
		const int nodeState = nodeStates[nodeIndex];

		// Valid node.
		if (nodeState != -1) {

			// New node index.
			int newNodeIndex = nodeOffsets[nodeIndex];
			newNodeIndex = nodeState > 0 ? newNodeIndex : newNodeIndex + newNumberOfInteriorNodes;

			// Parent.
			int nodeParentIndex = nodeParentIndices[nodeIndex];

			// Size.
			int nodeSize = nodeSizes[nodeIndex];

			// Child indices.
			int nodeLeftIndex = nodeLeftIndices[nodeIndex];
			int nodeRightIndex = nodeRightIndices[nodeIndex];

			// States.
			int nodeLeftState = nodeStates[nodeLeftIndex];
			int nodeRightState = nodeStates[nodeRightIndex];

			// Remap parent index.
			if (nodeParentIndex >= 0)
				nodeParentIndex = nodeOffsets[nodeParentIndex];

			// Boxes.
			Vec4f leftBoxMin, leftBoxMax;
			Vec4f rightBoxMin, rightBoxMax;

			// Interior
			if (nodeState > 0) {
				int nodeLeftOffset = nodeOffsets[nodeLeftIndex];
				int nodeRightOffset = nodeOffsets[nodeRightIndex];
				leftBoxMin = nodeBoxesMin[nodeLeftIndex];
				leftBoxMax = nodeBoxesMax[nodeLeftIndex];
				rightBoxMin = nodeBoxesMin[nodeRightIndex];
				rightBoxMax = nodeBoxesMax[nodeRightIndex];
				nodeLeftIndex = nodeLeftState > 0 ? nodeLeftOffset : ~(nodeLeftOffset + newNumberOfInteriorNodes);
				nodeRightIndex = nodeRightState > 0 ? nodeRightOffset : ~(nodeRightOffset + newNumberOfInteriorNodes);
			}

			// Leaf.
			else {
				int triangleOffset = triangleOffsets[nodeIndex];
				leftBoxMin = rightBoxMin = nodeBoxesMin[nodeIndex];
				leftBoxMax = rightBoxMax = nodeBoxesMax[nodeIndex];
				nodeLeftIndex = triangleOffset;
				nodeRightIndex = nodeLeftIndex + nodeSize;
				nodeSize = ~nodeSize;
			}

			// Output node.
			nodes[newNodeIndex] = CudaBVHNode(
				leftBoxMin.x, leftBoxMax.x, leftBoxMin.y, leftBoxMax.y,
				rightBoxMin.x, rightBoxMax.x, rightBoxMin.y, rightBoxMax.y,
				leftBoxMin.z, leftBoxMax.z, rightBoxMin.z, rightBoxMax.z,
				nodeLeftIndex, nodeRightIndex, nodeSize, nodeParentIndex
				);

		}

	}

}

extern "C" __global__ void reorderTriangles(
	const int threads,
	const int numberOfTriangles,
	int * triangleOffsets,
	int * inputTriangleIndices,
	int * outputTriangleIndices,
	int * leafIndices
	) {

	// Thread index.
	const int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

	for (int triangleIndex = threadIndex; triangleIndex < numberOfTriangles; triangleIndex += threads) {

		// Write triangle index.
		const int leafIndex = leafIndices[triangleIndex];
		const int triangleOffset = atomicAdd(&triangleOffsets[leafIndex], 1);
		outputTriangleIndices[triangleOffset] = inputTriangleIndices[triangleIndex];

	}

}

extern "C" __global__ void convert(
	const int threads,
	const int numberOfNodes,
	int * nodeParentIndices,
	int * nodeLeftIndices,
	int * nodeRightIndices,
	int * nodeSizes,
	Vec4f * nodeBoxesMin,
	Vec4f * nodeBoxesMax,
	CudaBVHNode * nodes
	) {

	// Thread index.
	const int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

	const int numberOfInteriorNodes = numberOfNodes >> 1;

	for (int nodeIndex = threadIndex; nodeIndex < numberOfNodes; nodeIndex += threads) {

		// Interior.
		bool interior = nodeIndex < numberOfInteriorNodes;

		// Size.
		int nodeSize = nodeSizes[nodeIndex];
		nodeSize = interior ? nodeSize : ~nodeSize;

		// Parent.
		int nodeParent = nodeParentIndices[nodeIndex];

		// Child indices.
		int nodeLeftIndex = nodeLeftIndices[nodeIndex];
		int nodeRightIndex = nodeRightIndices[nodeIndex];

		// Box.
		Vec4f leftBoxMin, leftBoxMax;
		Vec4f rightBoxMin, rightBoxMax;

		// Interior
		if (interior) {
			leftBoxMin = nodeBoxesMin[nodeLeftIndex];
			leftBoxMax = nodeBoxesMax[nodeLeftIndex];
			rightBoxMin = nodeBoxesMin[nodeRightIndex];
			rightBoxMax = nodeBoxesMax[nodeRightIndex];
			nodeLeftIndex = nodeLeftIndex < numberOfInteriorNodes ? nodeLeftIndex : ~nodeLeftIndex;
			nodeRightIndex = nodeRightIndex < numberOfInteriorNodes ? nodeRightIndex : ~nodeRightIndex;
		}

		// Leaf.
		else {
			leftBoxMin = rightBoxMin = nodeBoxesMin[nodeIndex];
			leftBoxMax = rightBoxMax = nodeBoxesMax[nodeIndex];
		}

		// Output node.
		nodes[nodeIndex] = CudaBVHNode(
			leftBoxMin.x, leftBoxMax.x, leftBoxMin.y, leftBoxMax.y,
			rightBoxMin.x, rightBoxMax.x, rightBoxMin.y, rightBoxMax.y,
			leftBoxMin.z, leftBoxMax.z, rightBoxMin.z, rightBoxMax.z,
			nodeLeftIndex, nodeRightIndex, nodeSize, nodeParent
			);

	}

}
