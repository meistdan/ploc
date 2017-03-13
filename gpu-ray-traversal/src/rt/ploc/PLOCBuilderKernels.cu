/**
* \file		PLOCBuilderKernels.cu
* \author	Daniel Meister
* \date		2017/01/23
* \brief	PLOCBuilder kernels soruce file.
*/

#include "PLOCBuilderKernels.h"
#include "CudaBVHUtil.cuh"

using namespace FW;

extern "C" __global__ void computeSceneBox(
	const int threads,
	const int numberOfVertices
) {

	// Thread index.
	const int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

	// Bounding box within the thread.
	AABB box; Vec3f vertex;
	for (int vertexIndex = threadIndex; vertexIndex < numberOfVertices; vertexIndex += threads) {
		vertexFromTexture(vertexIndex, vertex);
		box.grow(vertex);
	}

	// Cache.
	__shared__ float cache[3 * PLOC_REDUCTION_BLOCK_THREADS];
	Vec3f * bound = (Vec3f*)cache;

	// Min.
	bound[threadIdx.x] = box.min();
	bound[threadIdx.x] = min(bound[threadIdx.x], bound[threadIdx.x ^ 1]);
	bound[threadIdx.x] = min(bound[threadIdx.x], bound[threadIdx.x ^ 2]);
	bound[threadIdx.x] = min(bound[threadIdx.x], bound[threadIdx.x ^ 4]);
	bound[threadIdx.x] = min(bound[threadIdx.x], bound[threadIdx.x ^ 8]);
	bound[threadIdx.x] = min(bound[threadIdx.x], bound[threadIdx.x ^ 16]);

	__syncthreads();
	if ((threadIdx.x & 32) == 0) bound[threadIdx.x] = min(bound[threadIdx.x], bound[threadIdx.x ^ 32]);

	__syncthreads();
	if ((threadIdx.x & 64) == 0) bound[threadIdx.x] = min(bound[threadIdx.x], bound[threadIdx.x ^ 64]);

	__syncthreads();
	if ((threadIdx.x & 128) == 0) bound[threadIdx.x] = min(bound[threadIdx.x], bound[threadIdx.x ^ 128]);

	// Update global bounding box.
	if (threadIdx.x == 0) {
		atomicMin(&sceneBox[0], bound[threadIdx.x].x);
		atomicMin(&sceneBox[1], bound[threadIdx.x].y);
		atomicMin(&sceneBox[2], bound[threadIdx.x].z);
	}

	// Max.
	bound[threadIdx.x] = box.max();
	bound[threadIdx.x] = max(bound[threadIdx.x], bound[threadIdx.x ^ 1]);
	bound[threadIdx.x] = max(bound[threadIdx.x], bound[threadIdx.x ^ 2]);
	bound[threadIdx.x] = max(bound[threadIdx.x], bound[threadIdx.x ^ 4]);
	bound[threadIdx.x] = max(bound[threadIdx.x], bound[threadIdx.x ^ 8]);
	bound[threadIdx.x] = max(bound[threadIdx.x], bound[threadIdx.x ^ 16]);

	__syncthreads();
	if ((threadIdx.x & 32) == 0) bound[threadIdx.x] = max(bound[threadIdx.x], bound[threadIdx.x ^ 32]);

	__syncthreads();
	if ((threadIdx.x & 64) == 0) bound[threadIdx.x] = max(bound[threadIdx.x], bound[threadIdx.x ^ 64]);

	__syncthreads();
	if ((threadIdx.x & 128) == 0) bound[threadIdx.x] = max(bound[threadIdx.x], bound[threadIdx.x ^ 128]);

	// Update global bounding box.
	if (threadIdx.x == 0) {
		atomicMax(&sceneBox[3], bound[threadIdx.x].x);
		atomicMax(&sceneBox[4], bound[threadIdx.x].y);
		atomicMax(&sceneBox[5], bound[threadIdx.x].z);
	}

}

extern "C" __global__ void computeMortonCodes30(
	const int threads,
	const int numberOfTriangles,
	Vec4f * nodeBoxesMin,
	Vec4f * nodeBoxesMax,
	int * triangleIndices,
	unsigned int * mortonCodes
	) {

	// Thread index.
	const int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

	// Scene box.
	AABB _sceneBox = *(AABB*)sceneBoxConst;
	Vec3f scale = 1.0f / (_sceneBox.max() - _sceneBox.min());

	for (int triangleIndex = threadIndex; triangleIndex < numberOfTriangles; triangleIndex += threads) {

		// Triangle.		
		Vec3f v0, v1, v2;
		verticesFromTexture(triangleIndex, v0, v1, v2);

		// Box.
		AABB box;
		box.grow(v0);
		box.grow(v1);
		box.grow(v2);

		// Node box
		const int nodeIndex = triangleIndex + numberOfTriangles - 1;
		nodeBoxesMin[nodeIndex] = Vec4f(box.min(), 0.0f);
		nodeBoxesMax[nodeIndex] = Vec4f(box.max(), 0.0f);

		// Triangle index, node index and Morton code.
		triangleIndices[triangleIndex] = triangleIndex;
		mortonCodes[triangleIndex] = mortonCode((box.midPoint() - _sceneBox.min()) * scale);

	}

}

extern "C" __global__ void computeMortonCodes60(
	const int threads,
	const int numberOfTriangles,
	Vec4f * nodeBoxesMin,
	Vec4f * nodeBoxesMax,
	int * triangleIndices,
	unsigned long long * mortonCodes
	) {

	// Thread index.
	const int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

	// Scene box.
	AABB _sceneBox = *(AABB*)sceneBoxConst;
	Vec3f scale = 1.0f / (_sceneBox.max() - _sceneBox.min());

	for (int triangleIndex = threadIndex; triangleIndex < numberOfTriangles; triangleIndex += threads) {

		// Triangle.		
		Vec3f v0, v1, v2;
		verticesFromTexture(triangleIndex, v0, v1, v2);

		// Box.
		AABB box;
		box.grow(v0);
		box.grow(v1);
		box.grow(v2);

		// Node box
		const int nodeIndex = triangleIndex + numberOfTriangles - 1;
		nodeBoxesMin[nodeIndex] = Vec4f(box.min(), 0.0f);
		nodeBoxesMax[nodeIndex] = Vec4f(box.max(), 0.0f);

		// Triangle index, node index and Morton code.
		triangleIndices[triangleIndex] = triangleIndex;
		mortonCodes[triangleIndex] = mortonCode64((box.midPoint() - _sceneBox.min()) * scale);

	}

}

extern "C" __global__ void setupClusters(
	const int threads,
	const int numberOfTriangles,
	int * nodeLeftIndices,
	int * nodeRightIndices,
	int * nodeSizes,
	int * nodeIndices
	) {

	// Thread index.
	const int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

	for (int triangleIndex = threadIndex; triangleIndex < numberOfTriangles; triangleIndex += threads) {

		// Node.
		const int nodeIndex = triangleIndex + numberOfTriangles - 1;
		nodeLeftIndices[nodeIndex] = triangleIndex;
		nodeRightIndices[nodeIndex] = triangleIndex + 1;
		nodeSizes[nodeIndex] = 1;

		// Node index.
		nodeIndices[triangleIndex] = nodeIndex;

	}

}

extern "C" __global__ void generateNeighboursCached(
	const int numberOfClusters,
	const int radius,
	float * neighbourDistances,
	int * neighbourIndices,
	Vec4f * nodeBoxesMin,
	Vec4f * nodeBoxesMax,
	int * nodeIndices
	) {

	// Shared memory cache.
	__shared__ char cache[sizeof(AABB)* 2 * PLOC_GEN_BLOCK_THREADS];
	AABB * boxes = ((AABB*)cache) + blockDim.x / 2;

	// Thread index.
	const int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

	// Block offset.
	const int blockOffset = blockDim.x * blockIdx.x;

	// Load boxes.
	for (int neighbourIndex = int(threadIdx.x) - radius; neighbourIndex < int(blockDim.x) + radius; neighbourIndex += blockDim.x) {

		// Cluster index.
		int clusterIndex = neighbourIndex + blockOffset;

		// Valid threads.
		if (clusterIndex >= 0 && clusterIndex < numberOfClusters) {

			// Node index.
			int nodeIndex = nodeIndices[clusterIndex];

			// Cluster bounding box.
			const Vec4f boxMin = nodeBoxesMin[nodeIndex];
			const Vec4f boxMax = nodeBoxesMax[nodeIndex];
			boxes[neighbourIndex] = AABB(Vec3f(boxMin.x, boxMin.y, boxMin.z), Vec3f(boxMax.x, boxMax.y, boxMax.z));

		}

		// Dummy large boxes.
		else {
			boxes[neighbourIndex] = AABB(Vec3f(-FW_F32_MAX), Vec3f(FW_F32_MAX));
		}

	}

	// Sync.
	__syncthreads();

	// Nearest neighbour.
	int minIndex = -1;
	float minDistance = FW_F32_MAX;

	// Cluster box.
	AABB box = boxes[threadIdx.x];

	// Search left.
	for (int neighbourIndex = int(threadIdx.x) - radius; neighbourIndex < int(threadIdx.x); ++neighbourIndex) {

		// Box.
		AABB neighbourBox = boxes[neighbourIndex];

		// Grow.
		neighbourBox.grow(box);

		// Surface area.
		const float distance = neighbourBox.area();

		// Update distance.
		if (minDistance > distance) {
			minIndex = blockOffset + neighbourIndex;
			minDistance = distance;
		} /*else if (minDistance == distance) {
			minIndex = FW::min(minIndex, blockOffset + neighbourIndex);
		}*/

	}

	// Search right.
	for (int neighbourIndex = threadIdx.x + 1; neighbourIndex < threadIdx.x + radius + 1; ++neighbourIndex) {

		// Box.
		AABB neighbourBox = boxes[neighbourIndex];

		// Grow.
		neighbourBox.grow(box);

		// Surface area.
		const float distance = neighbourBox.area();

		// Update distance.
		if (minDistance > distance) {
			minIndex = blockOffset + neighbourIndex;
			minDistance = distance;
		} /*else if (minDistance == distance) {
			minIndex = FW::min(minIndex, blockOffset + neighbourIndex);
		}*/

	}

	// Save proposal.
	if (threadIndex < numberOfClusters) {
		const int nodeIndex = nodeIndices[threadIndex];
		neighbourDistances[nodeIndex] = minDistance;
		neighbourIndices[nodeIndex] = minIndex;
	} 

}

extern "C" __global__ void generateNeighbours(
	const int numberOfClusters,
	const int radius,
	float * neighbourDistances,
	int * neighbourIndices,
	Vec4f * nodeBoxesMin,
	Vec4f * nodeBoxesMax,
	int * nodeIndices
	) {

	// Thread index.
	const int clusterIndex = blockDim.x * blockIdx.x + threadIdx.x;

	if (clusterIndex < numberOfClusters) {

		// Node index.
		const int nodeIndex = nodeIndices[clusterIndex];

		// Cluster bounding box.
		const Vec4f boxMin = nodeBoxesMin[nodeIndex];
		const Vec4f boxMax = nodeBoxesMax[nodeIndex];
		AABB box = AABB(Vec3f(boxMin.x, boxMin.y, boxMin.z), Vec3f(boxMax.x, boxMax.y, boxMax.z));

		// Nearest neighbour.
		int minIndex = -1;
		float minDistance = FW_F32_MAX;

		// Search left.
		for (int neighbourIndex = FW::max(0, clusterIndex - radius); neighbourIndex < clusterIndex; ++neighbourIndex) {

			// Neighbour node index.
			const int neighbourNodeIndex = nodeIndices[neighbourIndex];

			// Box.
			Vec4f neighbourBoxMin = nodeBoxesMin[neighbourNodeIndex];
			Vec4f neighbourBoxMax = nodeBoxesMax[neighbourNodeIndex];
			AABB neighbourBox = AABB(Vec3f(neighbourBoxMin.x, neighbourBoxMin.y, neighbourBoxMin.z), 
									 Vec3f(neighbourBoxMax.x, neighbourBoxMax.y, neighbourBoxMax.z));
			neighbourBox.grow(box);

			// Surface area.
			const float distance = neighbourBox.area();

			// Update distance.
			if (minDistance > distance) {
				minDistance = distance;
				minIndex = neighbourIndex;
			}

		}

		// Search right.
		for (int neighbourIndex = clusterIndex + 1; neighbourIndex < FW::min(numberOfClusters, clusterIndex + radius + 1); ++neighbourIndex) {

			// Neighbour node index.
			const int neighbourNodeIndex = nodeIndices[neighbourIndex];

			// Box.
			Vec4f neighbourBoxMin = nodeBoxesMin[neighbourNodeIndex];
			Vec4f neighbourBoxMax = nodeBoxesMax[neighbourNodeIndex];
			AABB neighbourBox = AABB(Vec3f(neighbourBoxMin.x, neighbourBoxMin.y, neighbourBoxMin.z),
									 Vec3f(neighbourBoxMax.x, neighbourBoxMax.y, neighbourBoxMax.z));
			neighbourBox.grow(box);

			// Surface area.
			const float distance = neighbourBox.area();

			// Update distance.
			if (minDistance > distance) {
				minDistance = distance;
				minIndex = neighbourIndex;
			}

		}

		// Save proposal.
		neighbourDistances[nodeIndex] = minDistance;
		neighbourIndices[nodeIndex] = minIndex;

	}

}

extern "C" __global__ void merge(
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
	) {

	// Thread index.
	const int clusterIndex = blockDim.x * blockIdx.x + threadIdx.x;

	// Warp thread index.
	const int warpThreadIndex = threadIdx.x & (WARP_THREADS - 1);

	if (clusterIndex < numberOfClusters) {

		// Merging flag.
		bool merging = false;

		// Neighbour indices.
		const int leftNodeIndex = nodeIndices0[clusterIndex];
		const int neighbourIndex = neighbourIndices[leftNodeIndex];
		const int rightNodeIndex = nodeIndices0[neighbourIndex];
		const int neighbourNeighbourIndex = neighbourIndices[rightNodeIndex];

		// Merge only mutually paired clusters.
		if (clusterIndex == neighbourNeighbourIndex) {
			if (clusterIndex < neighbourIndex) merging = true;
		}

		// Just copy the node index.
		else {
			nodeIndices1[clusterIndex] = leftNodeIndex;
		}

		// Prefix scan.
		const unsigned int warpBallot = __ballot(merging);
		const int warpCount = __popc(warpBallot);
		const int warpIndex = __popc(warpBallot & ((1u << warpThreadIndex) - 1));

		// Add count of components to the global counter.
		int warpOffset;
		if (warpThreadIndex == 0)
			warpOffset = atomicAdd(&prefixScanOffset, warpCount);

		// Exchange offset between threads.
		warpOffset = __shfl(warpOffset, 0);

		// Node index.
		const int nodeIndex = nodeOffset - warpOffset - warpIndex;

		// Merge.
		if (merging) {

			// Box min.
			Vec4f leftNodeBoxMin = nodeBoxesMin[leftNodeIndex];
			Vec4f rightNodeBoxMin = nodeBoxesMin[rightNodeIndex];
			leftNodeBoxMin.x = fminf(leftNodeBoxMin.x, rightNodeBoxMin.x);
			leftNodeBoxMin.y = fminf(leftNodeBoxMin.y, rightNodeBoxMin.y);
			leftNodeBoxMin.z = fminf(leftNodeBoxMin.z, rightNodeBoxMin.z);
			nodeBoxesMin[nodeIndex] = leftNodeBoxMin;

			// Box max.
			Vec4f leftNodeBoxMax = nodeBoxesMax[leftNodeIndex];
			Vec4f rightNodeBoxMax = nodeBoxesMax[rightNodeIndex];
			leftNodeBoxMax.x = fmaxf(leftNodeBoxMax.x, rightNodeBoxMax.x);
			leftNodeBoxMax.y = fmaxf(leftNodeBoxMax.y, rightNodeBoxMax.y);
			leftNodeBoxMax.z = fmaxf(leftNodeBoxMax.z, rightNodeBoxMax.z);
			nodeBoxesMax[nodeIndex] = leftNodeBoxMax;

			// Children.
			const int nodeSize = nodeSizes[leftNodeIndex] + nodeSizes[rightNodeIndex];

			// Parent indices.
			nodeParentIndices[leftNodeIndex] = nodeIndex;
			nodeParentIndices[rightNodeIndex] = nodeIndex;

			// Node.
			nodeSizes[nodeIndex] = nodeSize;
			nodeLeftIndices[nodeIndex] = leftNodeIndex;
			nodeRightIndices[nodeIndex] = rightNodeIndex;

			// Update node index.
			nodeIndices1[clusterIndex] = nodeIndex;
			nodeIndices1[neighbourIndex] = -1;

		}

	}

}

extern "C" __global__ void localPrefixScan(
	const int numberOfClusters,
	int * nodeIndices,
	int * threadOffsets,
	int * blockOffsets
	) {

	// Thread index.
	const int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

	// Cache.
	__shared__ volatile int blockCache[2 * PLOC_SCAN_BLOCK_THREADS];

	// Read value.
	int threadValue = 0;
	if (threadIndex < numberOfClusters)
		threadValue = nodeIndices[threadIndex] >= 0;

	// Block scan.
	int blockSum = threadValue;
	blockScan<PLOC_SCAN_BLOCK_THREADS>(blockSum, blockCache);
	blockSum -= threadValue;

	// Write value.
	if (threadIndex < numberOfClusters)
		threadOffsets[threadIndex] = blockSum;

	// Write block value.
	if (threadIdx.x == 0)
		blockOffsets[blockIdx.x] = blockCache[2 * PLOC_SCAN_BLOCK_THREADS - 1];

}

extern "C" __global__ void globalPrefixScan(
	const int numberOfBlocks,
	int * blockOffsets
	) {

	// Block end.
	const int blockEnd = divCeil(numberOfBlocks, PLOC_SCAN_BLOCK_THREADS) * PLOC_SCAN_BLOCK_THREADS;

	// Cache.
	__shared__ volatile int blockCache[2 * PLOC_SCAN_BLOCK_THREADS];

	if (blockIdx.x == 0) {

		// Block offset.
		int blockOffset = 0;

		for (int blockIndex = threadIdx.x; blockIndex < blockEnd; blockIndex += PLOC_SCAN_BLOCK_THREADS) {

			// Read value.
			int blockValue = 0;
			if (blockIndex < numberOfBlocks)
				blockValue = blockOffsets[blockIndex];

			// Block scan.
			int blockSum = blockValue;
			blockScan<PLOC_SCAN_BLOCK_THREADS>(blockSum, blockCache);
			blockSum -= blockValue;

			// Write value.
			if (blockIndex < numberOfBlocks)
				blockOffsets[blockIndex] = blockSum + blockOffset;

			// Update block offset.
			blockOffset += blockCache[2 * PLOC_SCAN_BLOCK_THREADS - 1];

		}

	}

}

extern "C" __global__ void compact(
	const int numberOfClusters,
	int * nodeIndices0,
	int * nodeIndices1,
	int * blockOffsets,
	int * threadOffsets
	) {

	// Thread index.
	const int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

	// Only valid clusters.
	if (threadIndex < numberOfClusters) {

		// Compact.
		const int nodeIndex = nodeIndices0[threadIndex];
		const int newClusterIndex = blockOffsets[blockIdx.x] + threadOffsets[threadIndex];
		if (nodeIndex >= 0)
			nodeIndices1[newClusterIndex] = nodeIndex;

	}

}

extern "C" __global__ void woopifyTriangles(
	const int threads,
	const int numberOfTriangles,
	int * triangleIndices,
	Vec4f * triWoopsA,
	Vec4f * triWoopsB,
	Vec4f * triWoopsC
	) {

	// Thread index.
	const int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

	// Woop's matrix.
	Mat4f im;

	for (int triangleIndex = threadIndex; triangleIndex < numberOfTriangles; triangleIndex += threads) {

		// Triangle.
		Vec3f v0, v1, v2;
		verticesFromTexture(triangleIndices[triangleIndex], v0, v1, v2);

		// Woopify triangle.
		im.setCol(0, Vec4f(v0 - v2, 0.0f));
		im.setCol(1, Vec4f(v1 - v2, 0.0f));
		im.setCol(2, Vec4f(cross(v0 - v2, v1 - v2), 0.0f));
		im.setCol(3, Vec4f(v2, 1.0f));
		im = invert(im);

		triWoopsA[triangleIndex] = Vec4f(im(2, 0), im(2, 1), im(2, 2), -im(2, 3));
		triWoopsB[triangleIndex] = im.getRow(0);
		triWoopsC[triangleIndex] = im.getRow(1);

	}

}

extern "C" __global__ void computeCost(
	const int threads,
	const int numberOfNodes,
	const float sceneBoxArea,
	const float ct,
	const float ci,
	CudaBVHNode * nodes
) {

	// Thread index.
	const int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;

	// Cost.
	float _cost = 0.0f;

	for (int nodeIndex = threadIndex; nodeIndex < numberOfNodes; nodeIndex += threads) {

		CudaBVHNode node = nodes[nodeIndex];
		float P = node.getSurfaceArea() / sceneBoxArea;

		// Leaf.
		if (node.isLeaf()) {
			_cost += ci * P * node.getSize();
		}

		// Interior node.
		else {
			_cost += ct * P;
		}
	}

	// Cache.
	__shared__ volatile float cache[PLOC_REDUCTION_BLOCK_THREADS];

	// Cost reduction.
	cache[threadIdx.x] = _cost;
	cache[threadIdx.x] += cache[threadIdx.x ^ 1];
	cache[threadIdx.x] += cache[threadIdx.x ^ 2];
	cache[threadIdx.x] += cache[threadIdx.x ^ 4];
	cache[threadIdx.x] += cache[threadIdx.x ^ 8];
	cache[threadIdx.x] += cache[threadIdx.x ^ 16];

	__syncthreads();
	if ((threadIdx.x & 32) == 0) cache[threadIdx.x] += cache[threadIdx.x ^ 32];

	__syncthreads();
	if ((threadIdx.x & 64) == 0) cache[threadIdx.x] += cache[threadIdx.x ^ 64];

	__syncthreads();
	if ((threadIdx.x & 128) == 0) cache[threadIdx.x] += cache[threadIdx.x ^ 128];

	// Update total cost.
	if (threadIdx.x == 0) {
		atomicAdd(&cost, cache[threadIdx.x]);
	}

}
