/**
* \file	CudaBVHNode.h
* \author	Daniel Meister
* \date	2016/02/23
* \brief	CudaBVHNode struct header file.
*/

#ifndef _CUDA_BVH_NODE_H_
#define _CUDA_BVH_NODE_H_

#include "base/DLLImports.hpp"
#include "Util.hpp"

using namespace FW;

struct CudaBVHNode {

	float leftMinX;
	float leftMaxX;
	float leftMinY;
	float leftMaxY;
	float rightMinX;
	float rightMaxX;
	float rightMinY;
	float rightMaxY;
	float leftMinZ;
	float leftMaxZ;
	float rightMinZ;
	float rightMaxZ;
	int begin;
	int end;
	int size;
	int parent;

	FW_CUDA_FUNC CudaBVHNode(void) {}

	FW_CUDA_FUNC CudaBVHNode(
		float leftMinX,
		float leftMaxX,
		float leftMinY,
		float leftMaxY,
		float rightMinX,
		float rightMaxX,
		float rightMinY,
		float rightMaxY,
		float leftMinZ,
		float leftMaxZ,
		float rightMinZ,
		float rightMaxZ,
		int begin,
		int end,
		int size,
		int parent
		) :
		leftMinX(leftMinX),
		leftMaxX(leftMaxX),
		leftMinY(leftMinY),
		leftMaxY(leftMaxY),
		rightMinX(rightMinX),
		rightMaxX(rightMaxX),
		rightMinY(rightMinY),
		rightMaxY(rightMaxY),
		leftMinZ(leftMinZ),
		leftMaxZ(leftMaxZ),
		rightMinZ(rightMinZ),
		rightMaxZ(rightMaxZ),
		begin(begin),
		end(end),
		size(size),
		parent(parent) {}

	FW_CUDA_FUNC bool isLeaf(void) {
		return size < 0;
	}

	FW_CUDA_FUNC int getSize(void) {
		return size < 0 ? ~size : size;
	}

	FW_CUDA_FUNC int getParentIndex(void) {
		return parent;
	}

	FW_CUDA_FUNC AABB getLeftBoundingBox(void) {
		AABB box;
		box.min() = Vec3f(leftMinX, leftMinY, leftMinZ);
		box.max() = Vec3f(leftMaxX, leftMaxY, leftMaxZ);
		return box;
	}

	FW_CUDA_FUNC AABB getRightBoundingBox(void) {
		AABB box;
		box.min() = Vec3f(rightMinX, rightMinY, rightMinZ);
		box.max() = Vec3f(rightMaxX, rightMaxY, rightMaxZ);
		return box;
	}

	FW_CUDA_FUNC AABB getBoundingBox(void) {
		AABB box;
		AABB leftBox = getLeftBoundingBox();
		AABB rightBox = getRightBoundingBox();
		box.min() = min(leftBox.min(), rightBox.min());
		box.max() = max(leftBox.max(), rightBox.max());
		return box;
	}

	FW_CUDA_FUNC float getSurfaceArea(void) {
		return getBoundingBox().area();
	}

};

#endif /* _CUDA_BVH_NODE_H_ */
