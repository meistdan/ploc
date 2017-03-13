/**
* \file		PLOCBuilder.cpp
* \author	Daniel Meister
* \date		2017/01/23
* \brief	PLOCBuilder class source file.
*/

#include "PLOCBuilder.h"
#include "BVHUtil.h"
#include "Cub.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stack>

using namespace FW;

bool PLOCBuilder::validate(CudaBVH & bvh, Scene * scene) {

	bool valid = true;

	// Nodes.
	CudaBVHNode * nodes = (CudaBVHNode*)bvh.getNodeBuffer().getPtr();

	// Triangle indices.
	int * triangleIndices = (int*)bvh.getTriIndexBuffer().getPtr();

	// Nodes histogram.
	int numberOfNodes = bvh.getNodeBuffer().getSize() / sizeof(CudaBVHNode);
	std::vector<int> nodeHistogram(numberOfNodes);
	memset(nodeHistogram.data(), 0, sizeof(int)* numberOfNodes);
	nodeHistogram[0]++;

	// Triangle histogram.
	int numberOfTriangles = scene->getNumTriangles();
	std::vector<int> triangleHistogram(numberOfTriangles);
	memset(triangleHistogram.data(), 0, sizeof(int)* numberOfTriangles);

	// Check triangle indices.
	for (int i = 0; i < numberOfTriangles; ++i) {
		triangleHistogram[triangleIndices[i]]++;
	}

	for (int i = 0; i < numberOfTriangles; ++i) {
		if (triangleHistogram[i] < 1) {
			printf("<PLOCBuilder> Invalid triangle indices!\n");
			valid = false;
		}
	}

	// Reset triangle histogram.
	memset(triangleHistogram.data(), 0, sizeof(int)* numberOfTriangles);

	// Stack.
	std::stack<int> stack;
	stack.push(0);

	// Traverse BVH.
	while (!stack.empty()) {

		// Pop.
		int nodeIndex = stack.top();
		stack.pop();
		CudaBVHNode & node = nodes[nodeIndex];

		// Interior.
		if (!node.isLeaf()) {

			// Child indices.
			int leftIndex = node.begin < 0 ? ~node.begin : node.begin;
			int rightIndex = node.end < 0 ? ~node.end : node.end;

			// Child nodes.
			CudaBVHNode & left = nodes[leftIndex];
			CudaBVHNode & right = nodes[rightIndex];

			// Parent index.
			if (left.getParentIndex() != nodeIndex || right.getParentIndex() != nodeIndex) {
				printf("<PLOCBuilder> Invalid parent index!\n");
				valid = false;
			}

			// Check sizes.
			if (node.getSize() != left.getSize() + right.getSize()) {
				printf("<PLOCBuilder> Invalid node size!\n");
				valid = false;
			}

			// Update histogram.
			nodeHistogram[leftIndex]++;
			nodeHistogram[rightIndex]++;

			// Push.
			stack.push(leftIndex);
			stack.push(rightIndex);

		}

		// Leaf.
		else {

			// Check Bounds.
			if (node.begin >= node.end) {
				printf("<PLOCBuilder> Invalid leaf bounds [%d, %d]!\n", node.begin, node.end);
				valid = false;
			}

			// Update histogram.
			for (int i = node.begin; i < node.end; ++i) {
				int triangleIndex = triangleIndices[i];
				triangleHistogram[triangleIndex]++;
			}

		}

	}

	// Check node histogram.
	for (int i = 0; i < numberOfNodes; ++i) {
		if (nodeHistogram[i] != 1) {
			printf("<PLOCBuilder> Not all nodes are referenced!\n");
			valid = false;
		}
	}

	// Check triangle histogram.
	for (int i = 0; i < numberOfTriangles; ++i) {
		if (triangleHistogram[i] < 1) {
			printf("<CudaBVH> Not all triangles are referenced!\n");
			valid = false;
		}
	}

	return valid;

}

void PLOCBuilder::configure() {

	const std::string configFile = "ploc.cfg";
	
	std::ifstream file(configFile);
	std::string line, key, value;

	int radiusTmp;
	int maxLeafSizeTmp;

	if (file.is_open()) {
		while (getline(file, line)) {
			std::istringstream ss(line);
			getline(ss, key, '=');
			getline(ss, value, '=');

			if (key == "radius") {
				std::istringstream(value) >> radiusTmp;
				setRadius(radiusTmp);
			}
			else if (key == "morton60") {
				std::istringstream(value) >> mortonCodes60Bits;
			}
			else if (key == "adaptiveLeafSize") {
				std::istringstream(value) >> adaptiveLeafSize;
			}
			else if (key == "maxLeafSize") {
				std::istringstream(value) >> maxLeafSizeTmp;
				setMaxLeafSize(maxLeafSizeTmp);
			}

		}
		file.close();
	}

}

void PLOCBuilder::allocate(int numberOfTriangles) {
	if (mortonCodes60Bits) {
		mortonCodes[0].resizeDiscard(sizeof(unsigned long long) * numberOfTriangles);
		mortonCodes[1].resizeDiscard(sizeof(unsigned long long) * numberOfTriangles);
	} else {
		mortonCodes[0].resizeDiscard(sizeof(unsigned int) * numberOfTriangles);
		mortonCodes[1].resizeDiscard(sizeof(unsigned int) * numberOfTriangles);
	}
	int maxNumberOfBlocks = divCeil(numberOfTriangles, PLOC_SCAN_BLOCK_THREADS);
	threadOffsets.resizeDiscard(sizeof(int)* maxNumberOfBlocks * PLOC_SCAN_BLOCK_THREADS);
	blockOffsets.resizeDiscard(sizeof(int) * maxNumberOfBlocks);
	nodeSizes.resizeDiscard(sizeof(int)* (2 * numberOfTriangles - 1));
	nodeLeftIndices.resizeDiscard(sizeof(int)* (2 * numberOfTriangles - 1));
	nodeRightIndices.resizeDiscard(sizeof(int)* (2 * numberOfTriangles - 1));
	nodeParentIndices.resizeDiscard(sizeof(int)* (2 * numberOfTriangles - 1));
	nodeBoxesMin.resizeDiscard(sizeof(Vec4f) * (2 * numberOfTriangles - 1));
	nodeBoxesMax.resizeDiscard(sizeof(Vec4f) * (2 * numberOfTriangles - 1));
	nodeIndices[0].resizeDiscard(sizeof(int)* numberOfTriangles);
	nodeIndices[1].resizeDiscard(sizeof(int)* numberOfTriangles);
	neighbourDistances.resizeDiscard(sizeof(float)* (2 * numberOfTriangles - 1));
	neighbourIndices.resizeDiscard(sizeof(int)* (2 * numberOfTriangles - 1));
	triangleIndices.resizeDiscard(sizeof(int)* numberOfTriangles);
}

float PLOCBuilder::computeSceneBox(Scene * scene) {

	// Kernel.
	CudaModule * module = compiler.compile();
	CudaKernel kernel = module->getKernel("computeSceneBox");

	// Texture memory.
	module->setTexRef("vertexTex", scene->getVtxPosBuffer(), CU_AD_FORMAT_FLOAT, 1);

	// Scene box.
	*(AABB*)module->getGlobal("sceneBox").getMutablePtr() = AABB();

	// Threads and blocks.
	int blockThreads = PLOC_REDUCTION_BLOCK_THREADS;
	int threads = scene->getNumVertices();

	// Set params.
	kernel.setParams(
		threads,
		scene->getNumVertices()
	);

	// Launch.
	float time = kernel.launchTimed(threads, Vec2i(blockThreads, 1));

	// Kernel time.
	return time;

}

float PLOCBuilder::computeMortonCodes(Scene * scene) {

	// Kernel.
	CudaModule * module = compiler.compile();
	CudaKernel kernel = module->getKernel(!mortonCodes60Bits ? "computeMortonCodes30" : "computeMortonCodes60");

	// Texture memory.
	module->setTexRef("triangleTex", scene->getTriVtxIndexBuffer(), CU_AD_FORMAT_UNSIGNED_INT32, 1);
	module->setTexRef("vertexTex", scene->getVtxPosBuffer(), CU_AD_FORMAT_FLOAT, 1);

	// Scene box.
	AABB sceneBox = *(AABB*)module->getGlobal("sceneBox").getPtr();
	Vec3f diag = sceneBox.max() - sceneBox.min();
	float edge = max(max(diag.x, diag.y), diag.z);
	sceneBox.max() = sceneBox.min() + Vec3f(edge);
	*(AABB*)module->getGlobal("sceneBoxConst").getMutablePtr() = sceneBox;

	// Threads and blocks.
	int blockThreads = PLOC_BLOCK_THREADS;
	int threads = scene->getNumTriangles();

	// Set params.
	kernel.setParams(
		threads,
		scene->getNumTriangles(),
		nodeBoxesMin,
		nodeBoxesMax,
		triangleIndices,
		mortonCodes[0]
		);

	// Launch.
	float time = kernel.launchTimed(threads, Vec2i(blockThreads, 1));

	// Kernel time.
	return time;

}

float PLOCBuilder::setupClusters(Scene * scene) {

	// Kernel.
	CudaModule * module = compiler.compile();
	CudaKernel kernel = module->getKernel("setupClusters");

	// Threads and blocks.
	int blockThreads = PLOC_BLOCK_THREADS;
	int threads = scene->getNumTriangles();

	// Set params.
	kernel.setParams(
		threads,
		scene->getNumTriangles(),
		nodeLeftIndices,
		nodeRightIndices,
		nodeSizes,
		nodeIndices[0]
	);

	// Launch.
	float time = kernel.launchTimed(threads, Vec2i(blockThreads, 1));

	// Kernel time.
	return time;

}

float PLOCBuilder::sortClusters(int numberOfClusters) {
	int * values0 = (int*)nodeIndices[0].getMutableCudaPtr();
	int * values1 = (int*)nodeIndices[1].getMutableCudaPtr();
	if (mortonCodes60Bits) {
		unsigned long long * keys0 = (unsigned long long*)mortonCodes[0].getMutableCudaPtr();
		unsigned long long * keys1 = (unsigned long long*)mortonCodes[1].getMutableCudaPtr();
		return Cub::sort(numberOfClusters, keys0, keys1, values0, values1, sortSwap);
	} else {
		unsigned int * keys0 = (unsigned int*)mortonCodes[0].getMutableCudaPtr();
		unsigned int * keys1 = (unsigned int*)mortonCodes[1].getMutableCudaPtr();
		return Cub::sort(numberOfClusters, keys0, keys1, values0, values1, sortSwap);
	}
}

float PLOCBuilder::clustering(int numberOfTriangles) {

	// Kernel time.
	float generateNeighboursTime = 0.0f;
	float mergeTime = 0.0f;
	float localPrefixScanTime = 0.0f;
	float globalPrefixScanTime = 0.0f;
	float compactTime = 0.0f;

	// Kernels.
	CudaModule * module = compiler.compile();
	CudaKernel generateNeighboursCachedKernel = module->getKernel("generateNeighboursCached");
	CudaKernel generateNeighboursKernel = module->getKernel("generateNeighbours");
	CudaKernel mergeKernel = module->getKernel("merge");
	CudaKernel localPrefixScanKernel = module->getKernel("localPrefixScan");
	CudaKernel globalPrefixScanKernel = module->getKernel("globalPrefixScan");
	CudaKernel compactKernel = module->getKernel("compact");

	// Threads and blocks.
	int mergeBlockThreads = PLOC_BLOCK_THREADS;

	// Number of clusters.
	int numberOfClusters = numberOfTriangles;

	// Swap flag.
	bool swapBuffers = sortSwap;

	// Step counter.
	steps = 0;

	// Main loop.
	while (numberOfClusters > 1) {

		// Increment step counter.
		++steps;

		// Generate neighbours.
		if (radius <= PLOC_GEN_BLOCK_THREADS / 2) {
			generateNeighboursCachedKernel.setParams(
				numberOfClusters,
				radius,
				neighbourDistances,
				neighbourIndices,
				nodeBoxesMin,
				nodeBoxesMax,
				nodeIndices[swapBuffers]
			);
			generateNeighboursTime += generateNeighboursCachedKernel.launchTimed(numberOfClusters, Vec2i(PLOC_GEN_BLOCK_THREADS, 1));
		}
		else {
			generateNeighboursKernel.setParams(
				numberOfClusters,
				radius,
				neighbourDistances,
				neighbourIndices,
				nodeBoxesMin,
				nodeBoxesMax,
				nodeIndices[swapBuffers]
			);
			generateNeighboursTime += generateNeighboursKernel.launchTimed(numberOfClusters, Vec2i(PLOC_GEN_BLOCK_THREADS, 1));
		}

		// Clear prefix scan offset.
		module->getGlobal("prefixScanOffset").clear();

		// Node offset.
		int nodeOffset = numberOfClusters - 2;

		// Merge.
		mergeKernel.setParams(
			numberOfClusters,
			nodeOffset,
			neighbourIndices,
			nodeIndices[swapBuffers],
			nodeIndices[!swapBuffers],
			nodeParentIndices,
			nodeLeftIndices,
			nodeRightIndices,
			nodeSizes,
			nodeBoxesMin,
			nodeBoxesMax
		);
		mergeTime += mergeKernel.launchTimed(numberOfClusters, Vec2i(mergeBlockThreads, 1));

		// New number of clusters.
		int newNumberOfClusters = numberOfClusters - *(int*)module->getGlobal("prefixScanOffset").getPtr();
		
		// Swap buffers.
		swapBuffers = !swapBuffers;

		// Local prefix scan.
		int numberOfBlocks = divCeil(numberOfClusters, PLOC_SCAN_BLOCK_THREADS);
		localPrefixScanKernel.setParams(
			numberOfClusters,
			nodeIndices[swapBuffers],
			threadOffsets,
			blockOffsets
		);
		localPrefixScanTime += localPrefixScanKernel.launchTimed(numberOfBlocks * PLOC_SCAN_BLOCK_THREADS, Vec2i(PLOC_SCAN_BLOCK_THREADS, 1));

		// Global prefix scan.
		globalPrefixScanKernel.setParams(
			numberOfBlocks,
			blockOffsets
			);
		globalPrefixScanTime += globalPrefixScanKernel.launchTimed(PLOC_SCAN_BLOCK_THREADS, Vec2i(PLOC_SCAN_BLOCK_THREADS, 1));

		// Compact.
		compactKernel.setParams(
			numberOfClusters,
			nodeIndices[swapBuffers],
			nodeIndices[!swapBuffers],
			blockOffsets,
			threadOffsets
		);
		compactTime += compactKernel.launchTimed(numberOfClusters, Vec2i(PLOC_SCAN_BLOCK_THREADS, 1));

		// Update number of clusters.
		numberOfClusters = newNumberOfClusters;

		// Swap buffers.
		swapBuffers = !swapBuffers;

	}

	// Kernels time.
	return generateNeighboursTime + mergeTime + localPrefixScanTime + globalPrefixScanTime + compactTime;

}

float PLOCBuilder::woopifyTriangles(CudaBVH & bvh, Scene * scene) {

	// Kernel.
	CudaModule * module = compiler.compile();
	CudaKernel kernel = module->getKernel("woopifyTriangles");

	// Texture memory.
	module->setTexRef("triangleTex", scene->getTriVtxIndexBuffer(), CU_AD_FORMAT_UNSIGNED_INT32, 1);
	module->setTexRef("vertexTex", scene->getVtxPosBuffer(), CU_AD_FORMAT_FLOAT, 1);

	// Threads and blocks.
	int blockThreads = PLOC_BLOCK_THREADS;
	int threads = scene->getNumTriangles();

	// Woop buffer.
	CUdeviceptr triPtr = bvh.getTriWoopBuffer().getCudaPtr();

	// Woop offsets.
	Vec2i triOfsA = bvh.getTriWoopSubArray(0);
	Vec2i triOfsB = bvh.getTriWoopSubArray(1);
	Vec2i triOfsC = bvh.getTriWoopSubArray(2);

	// Set params.
	kernel.setParams(
		threads,
		scene->getNumTriangles(),
		bvh.getTriIndexBuffer(),
		triPtr + triOfsA.x,
		triPtr + triOfsB.x,
		triPtr + triOfsC.x
		);

	// Launch.
	float time = kernel.launchTimed(threads, Vec2i(blockThreads, 1));

	// Kernel time.
	return time;

}

float PLOCBuilder::computeCost(CudaBVH & bvh) {

	// Kernel.
	CudaModule * module = compiler.compile();
	CudaKernel kernel = module->getKernel("computeCost");

	// Number of nodes.
	int numberOfNodes = int(bvh.getNodeBuffer().getSize() / sizeof(CudaBVHNode));

	// Threads and blocks.
	int blockThreads = PLOC_REDUCTION_BLOCK_THREADS;
	int threads = numberOfNodes;

	// Scee box.
	AABB sceneBox = *(AABB*)module->getGlobal("sceneBox").getPtr();

	// Reset cost.
	*(float*)module->getGlobal("cost").getMutablePtr() = 0.0f;

	// Set params.
	kernel.setParams(
		threads,
		numberOfNodes,
		sceneBox.area(),
		collapser.getCt(),
		collapser.getCi(),
		bvh.getNodeBuffer()
	);

	// Launch.
	kernel.launch(threads, Vec2i(blockThreads, 1));

	// Cost
	return *(float*)module->getGlobal("cost").getPtr();

}

float PLOCBuilder::build(CudaBVH & bvh, Scene * scene) {

	// Number of triangles.
	int numberOfTriangles = scene->getNumTriangles();

	// Allocate buffer.
	allocate(numberOfTriangles);

	// Compute scene box.
	float computeSceneBoxTime = computeSceneBox(scene);
	printf("<PLOCBuilder> Scene box computed in %fs.\n", computeSceneBoxTime);

	// Morton codes.
	float computeMortonCodesTime = computeMortonCodes(scene);
	printf("<PLOCBuilder> Morton codes (%d bits) computed in %fs.\n", mortonCodes60Bits ? 60 : 30, computeMortonCodesTime);

	// Setup clusters.
	float setupClustersTime = setupClusters(scene);
	printf("<PLOCBuilder> Clusters setup in %fs.\n", setupClustersTime);

	// Sort.
	float sortTime = sortClusters(numberOfTriangles);
	printf("<PLOCBuilder> Triangles sorted in %fs.\n", sortTime);

	// Clustering.
	float clusteringTime = clustering(numberOfTriangles);
	printf("<PLOCBuilder> Topology constructed in %fs.\n", clusteringTime);

	// Collapse.
	float collapseTime;
	if (adaptiveLeafSize)
		collapseTime = collapser.collapseAdaptive(numberOfTriangles, nodeSizes, nodeParentIndices,
		nodeLeftIndices, nodeRightIndices, nodeBoxesMin, nodeBoxesMax, triangleIndices, bvh);
	else
		collapseTime = collapser.collapse(numberOfTriangles, maxLeafSize, nodeSizes, nodeParentIndices,
		nodeLeftIndices, nodeRightIndices, nodeBoxesMin, nodeBoxesMax, triangleIndices, bvh);
	printf("<PLOCBuilder> BVH collapsed and converted in %fs.\n", collapseTime);

	// Woopify triangles.
	float woopTime = woopifyTriangles(bvh, scene);
	printf("<PLOCBuilder> Triangles woopified in %fs.\n", woopTime);

	return computeSceneBoxTime + computeMortonCodesTime + setupClustersTime + sortTime + clusteringTime + collapseTime + woopTime;

}

PLOCBuilder::PLOCBuilder() : radius(25), mortonCodes60Bits(true), maxLeafSize(8), adaptiveLeafSize(true) {
	compiler.setSourceFile("src/rt/ploc/PLOCBuilderKernels.cu");
	compiler.addOptions("-use_fast_math");
	compiler.include("src/rt");
	compiler.include("src/framework");
	configure();
}

PLOCBuilder::~PLOCBuilder() {
}

CudaBVH * PLOCBuilder::build(Scene * scene) {

	// Create BVH.
	CudaBVH * bvh = new CudaBVH(BVHLayout::BVHLayout_AOS_SOA);

	// Resize buffers.
	const int TRIANGLE_ALIGN = 4096;
	bvh->getNodeBuffer().resizeDiscard(sizeof(CudaBVHNode)* (2 * scene->getNumTriangles() - 1));
	bvh->getTriIndexBuffer().resizeDiscard(sizeof(int)* scene->getNumTriangles());
	bvh->getTriWoopBuffer().resizeDiscard((4 * sizeof(Vec4f) * scene->getNumTriangles() + TRIANGLE_ALIGN - 1) & -TRIANGLE_ALIGN);

	// Settings.
	printf("<PLOCBuilder> Radius %d, Morton codes %d bits.\n", radius, mortonCodes60Bits ? 60 : 30);
	printf("<PLOCBuilder> Adaptive collapse %d, Max. leaf size %d.\n", int(adaptiveLeafSize), maxLeafSize);

	// Build.
	float time = build(*bvh, scene);

	// Cost.
	float cost = computeCost(*bvh);

#if VALIDATE_BVH
	// Validate.
	validate(*bvh, scene);
#endif

	// Stats.
	printf("<PLOCBuilder> BVH built in %fs from %d triangles in %d steps.\n", time, scene->getNumTriangles(), steps);
	printf("<PLOCBuilder> %f MTriangles/s.\n", (scene->getNumTriangles() * 1.0e-3f / time), scene->getNumTriangles());
	printf("<PLOCBuilder> BVH cost is %f.\n", cost);

	return bvh;

}

int PLOCBuilder::getRadius() {
	return radius;
}

void PLOCBuilder::setRadius(int radius) {
	if (radius >= 1 && radius <= 128) this->radius = radius;
}

bool PLOCBuilder::isMortonCodes60Bits() {
	return mortonCodes60Bits;
}

void PLOCBuilder::setMortonCodes60Bits(bool mortonCodes60bits) {
	this->mortonCodes60Bits = mortonCodes60bits;
}

bool PLOCBuilder::getAdaptiveLeafSize() {
	return adaptiveLeafSize;
}

void PLOCBuilder::setAdaptiveLeafSize(bool adaptiveLeafSize) {
	this->adaptiveLeafSize = adaptiveLeafSize;
}

int PLOCBuilder::getMaxLeafSize() {
	return maxLeafSize;
}

void PLOCBuilder::setMaxLeafSize(int maxLeafSize) {
	if (maxLeafSize > 0 && maxLeafSize <= 64) this->maxLeafSize = maxLeafSize;
}
