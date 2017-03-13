/**
* \file		PLOCBuilder.h
* \author	Daniel Meister
* \date		2017/01/23
* \brief	PLOCBuilder class header file.
*/

#ifndef _PLOC_BUILDER_H_
#define _PLOC_BUILDER_H_

#include "gpu/CudaCompiler.hpp"
#include "cuda/CudaBVH.hpp"
#include "gpu/Buffer.hpp"

#include "BVHCollapser.h"
#include "PLOCBuilderKernels.h"

#define VALIDATE_BVH 0

namespace FW {

class PLOCBuilder {

private:

	CudaCompiler compiler;
	BVHCollapser collapser;

	Buffer triangleIndices;
	Buffer mortonCodes[2];

	Buffer threadOffsets;
	Buffer blockOffsets;
	Buffer prefixScanOffset;

	Buffer nodeSizes;
	Buffer nodeParentIndices;
	Buffer nodeLeftIndices;
	Buffer nodeRightIndices;
	Buffer nodeBoxesMin;
	Buffer nodeBoxesMax;
	Buffer nodeIndices[2];

	Buffer neighbourDistances;
	Buffer neighbourIndices;

	bool sortSwap;
	int steps;

	bool adaptiveLeafSize;
	bool mortonCodes60Bits;
	int maxLeafSize;
	int radius;

	bool validate(CudaBVH & bvh, Scene * scene);

	void configure(void);
	void allocate(int numberOfTriangles);

	float computeSceneBox(Scene * scene);
	float computeMortonCodes(Scene * scene);
	float setupClusters(Scene * scene);
	float sortClusters(int numberOfClusters);
	float clustering(int numberOfTriangles);
	float woopifyTriangles(CudaBVH & bvh, Scene * scene);
	float computeCost(CudaBVH & bvh);
	float build(CudaBVH & bvh, Scene * scene);

public:

	PLOCBuilder(void);
	~PLOCBuilder(void);

	CudaBVH * build(Scene * scene);

	int getRadius(void);
	void setRadius(int radius);
	bool isMortonCodes60Bits(void);
	void setMortonCodes60Bits(bool mortonCodes60Bits);
	bool getAdaptiveLeafSize(void);
	void setAdaptiveLeafSize(bool adaptiveLeafSize);
	int getMaxLeafSize(void);
	void setMaxLeafSize(int maxLeafSize);

};

};

#endif /* _PLOC_BUILDER_H_ */
