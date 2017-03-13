/**
* \file		CudaBVHUtil.cuh
* \author	Daniel Meister
* \date		2017/01/23
* \brief	A header file containing useful Cuda functions.
*/

#ifndef _CUDA_UTIL_H_
#define _CUDA_UTIL_H_

#include "base/Math.hpp"
#include "BVHUtil.h"

texture<int, 1> triangleTex;
texture<float, 1> vertexTex;
texture<float, 1> normalTex;
texture<float, 1> pseudocolorTex;

namespace FW {

//---------------------------------------------------------------------------
// FLOAT ATOMIC MIN / MAX
//---------------------------------------------------------------------------
	
__device__ __forceinline__ void atomicMin(float * ptr, float value) {
	unsigned int curr = atomicAdd((unsigned int*)ptr, 0);
	while (value < __int_as_float(curr)) {
		unsigned int prev = curr;
		curr = atomicCAS((unsigned int *)ptr, curr, __float_as_int(value));
		if (curr == prev)
			break;
	}
}

__device__ __forceinline__ void atomicMax(float * ptr, float value) {
	unsigned int curr = atomicAdd((unsigned int*)ptr, 0);
	while (value > __int_as_float(curr)) {
		unsigned int prev = curr;
		curr = atomicCAS((unsigned int*)ptr, curr, __float_as_int(value));
		if (curr == prev)
			break;
	}
}

//---------------------------------------------------------------------------
// MORTON CODE
//---------------------------------------------------------------------------

__device__ __forceinline__ unsigned int mortonCode(unsigned int x, unsigned int y, unsigned int z) {
	x = (x | (x << 16)) & 0x030000FF;
	x = (x | (x << 8)) & 0x0300F00F;
	x = (x | (x << 4)) & 0x030C30C3;
	x = (x | (x << 2)) & 0x09249249;
	y = (y | (y << 16)) & 0x030000FF;
	y = (y | (y << 8)) & 0x0300F00F;
	y = (y | (y << 4)) & 0x030C30C3;
	y = (y | (y << 2)) & 0x09249249;
	z = (z | (z << 16)) & 0x030000FF;
	z = (z | (z << 8)) & 0x0300F00F;
	z = (z | (z << 4)) & 0x030C30C3;
	z = (z | (z << 2)) & 0x09249249;
	return x | (y << 1) | (z << 2);
}

__device__ __forceinline__ unsigned int mortonCode(const float3 & centroid, const float3 & sceneExtentInv) {
	unsigned int x = (centroid.x * sceneExtentInv.x) * 1023u;
	unsigned int y = (centroid.y * sceneExtentInv.y) * 1023u;
	unsigned int z = (centroid.z * sceneExtentInv.z) * 1023u;
	return mortonCode(x, y, z);
}

__device__ __forceinline__ unsigned int mortonCode(const Vec3f & centroid, const Vec3f & sceneExtentInv) {
	unsigned int x = (centroid.x * sceneExtentInv.x) * 1023u;
	unsigned int y = (centroid.y * sceneExtentInv.y) * 1023u;
	unsigned int z = (centroid.z * sceneExtentInv.z) * 1023u;
	return mortonCode(x, y, z);
}

__device__ __forceinline__ unsigned int mortonCode(const Vec3f & centroid) {
	unsigned int x = centroid.x * 1023u;
	unsigned int y = centroid.y * 1023u;
	unsigned int z = centroid.z * 1023u;
	return mortonCode(x, y, z);
}

__device__ __forceinline__ unsigned long long mortonCode64(unsigned int x, unsigned int y, unsigned int z) {
	unsigned int loX = x & 1023u;
	unsigned int loY = y & 1023u;
	unsigned int loZ = z & 1023u;
	unsigned int hiX = x >> 10u;
	unsigned int hiY = y >> 10u;
	unsigned int hiZ = z >> 10u;
	unsigned long long lo = mortonCode(loX, loY, loZ);
	unsigned long long hi = mortonCode(hiX, hiY, hiZ);
	return (hi << 30) | lo;
}

__device__ __forceinline__ unsigned long long mortonCode64(const float3 & centroid, const float3 & sceneExtentInv) {
	unsigned int scale = (1u << 20) - 1;
	unsigned int x = (centroid.x * sceneExtentInv.x) * scale;
	unsigned int y = (centroid.y * sceneExtentInv.y) * scale;
	unsigned int z = (centroid.z * sceneExtentInv.z) * scale;
	return mortonCode64(x, y, z);
}

__device__ __forceinline__ unsigned long long mortonCode64(const Vec3f & centroid, const Vec3f & sceneExtentInv) {
	unsigned int scale = (1u << 20) - 1;
	unsigned int x = (centroid.x * sceneExtentInv.x) * scale;
	unsigned int y = (centroid.y * sceneExtentInv.y) * scale;
	unsigned int z = (centroid.z * sceneExtentInv.z) * scale;
	return mortonCode64(x, y, z);
}

__device__ __forceinline__ unsigned long long mortonCode64(const Vec3f & centroid) {
	unsigned int scale = (1u << 20) - 1;
	unsigned int x = centroid.x * scale;
	unsigned int y = centroid.y * scale;
	unsigned int z = centroid.z * scale;
	return mortonCode64(x, y, z);
}

//---------------------------------------------------------------------------
// PREFIX SCAN
//---------------------------------------------------------------------------

// Hillis-Steele warp scan.
__device__ __forceinline__ int warpScan(int warpThreadIndex, int warpSum) {
	int warpValue = warpSum;
	warpValue = __shfl_up(warpSum, 1); if (warpThreadIndex >= 1) warpSum += warpValue;
	warpValue = __shfl_up(warpSum, 2); if (warpThreadIndex >= 2) warpSum += warpValue;
	warpValue = __shfl_up(warpSum, 4); if (warpThreadIndex >= 4) warpSum += warpValue;
	warpValue = __shfl_up(warpSum, 8); if (warpThreadIndex >= 8) warpSum += warpValue;
	warpValue = __shfl_up(warpSum, 16); if (warpThreadIndex >= 16) warpSum += warpValue;
	return warpSum;
}

// Hillis-Steele warp scan.
__device__ __forceinline__ void warpScan(int warpThreadIndex, int & warpSum, volatile int * warpCache) {
	volatile int * warpHalfCache = warpCache + 32;
	warpCache[warpThreadIndex] = 0;
	warpHalfCache[warpThreadIndex] = warpSum;
	warpSum += warpHalfCache[warpThreadIndex - 1]; warpHalfCache[warpThreadIndex] = warpSum;
	warpSum += warpHalfCache[warpThreadIndex - 2]; warpHalfCache[warpThreadIndex] = warpSum;
	warpSum += warpHalfCache[warpThreadIndex - 4]; warpHalfCache[warpThreadIndex] = warpSum;
	warpSum += warpHalfCache[warpThreadIndex - 8]; warpHalfCache[warpThreadIndex] = warpSum;
	warpSum += warpHalfCache[warpThreadIndex - 16]; warpHalfCache[warpThreadIndex] = warpSum;
}

// Hillis-Steele block scan.
template <int SCAN_BLOCK_THREADS>
__device__ __forceinline__ void blockScan(int & blockSum, volatile int * blockCache) {
	volatile int * blockHalfCache = blockCache + SCAN_BLOCK_THREADS;
	const int threadIndex = (int)threadIdx.x;
	blockCache[threadIndex] = 0;
	blockHalfCache[threadIndex] = blockSum;
	for (int i = 1; i < SCAN_BLOCK_THREADS; i <<= 1) {
		__syncthreads();
		blockSum += blockHalfCache[threadIndex - i];
		__syncthreads();
		blockHalfCache[threadIndex] = blockSum;
	}
	__syncthreads();
}

template <>
__device__ __forceinline__ void blockScan<1024>(int & blockSum, volatile int * blockCache) {
	volatile int * blockHalfCache = blockCache + 1024;
	const int threadIndex = (int)threadIdx.x;
	blockCache[threadIdx.x] = 0;
	blockHalfCache[threadIdx.x] = blockSum;
	__syncthreads();
	blockSum += blockHalfCache[threadIndex - 1]; __syncthreads(); blockHalfCache[threadIndex] = blockSum; __syncthreads();
	blockSum += blockHalfCache[threadIndex - 2]; __syncthreads(); blockHalfCache[threadIndex] = blockSum; __syncthreads();
	blockSum += blockHalfCache[threadIndex - 4]; __syncthreads(); blockHalfCache[threadIndex] = blockSum; __syncthreads();
	blockSum += blockHalfCache[threadIndex - 8]; __syncthreads(); blockHalfCache[threadIndex] = blockSum; __syncthreads();
	blockSum += blockHalfCache[threadIndex - 16]; __syncthreads(); blockHalfCache[threadIndex] = blockSum; __syncthreads();
	blockSum += blockHalfCache[threadIndex - 32]; __syncthreads(); blockHalfCache[threadIndex] = blockSum; __syncthreads();
	blockSum += blockHalfCache[threadIndex - 64]; __syncthreads(); blockHalfCache[threadIndex] = blockSum; __syncthreads();
	blockSum += blockHalfCache[threadIndex - 128]; __syncthreads(); blockHalfCache[threadIndex] = blockSum; __syncthreads();
	blockSum += blockHalfCache[threadIndex - 256]; __syncthreads(); blockHalfCache[threadIndex] = blockSum; __syncthreads();
	blockSum += blockHalfCache[threadIndex - 512]; __syncthreads(); blockHalfCache[threadIndex] = blockSum; __syncthreads();
}

//---------------------------------------------------------------------------
// DATA IN TEXTURE
//---------------------------------------------------------------------------

__device__ __forceinline__ void pseudocolorFromTexture(int index, Vec3f & pseudocolor) {
	pseudocolor.x = tex1Dfetch(pseudocolorTex, 3 * index);
	pseudocolor.y = tex1Dfetch(pseudocolorTex, 3 * index + 1);
	pseudocolor.z = tex1Dfetch(pseudocolorTex, 3 * index + 2);
}

__device__ __forceinline__ void vertexFromTexture(int index, Vec3f & vertex) {
	vertex.x = tex1Dfetch(vertexTex, 3 * index);
	vertex.y = tex1Dfetch(vertexTex, 3 * index + 1);
	vertex.z = tex1Dfetch(vertexTex, 3 * index + 2);
}

__device__ __forceinline__ void normalFromTexture(int index, Vec3f & normal) {
	normal.x = tex1Dfetch(normalTex, 3 * index);
	normal.y = tex1Dfetch(normalTex, 3 * index + 1);
	normal.z = tex1Dfetch(normalTex, 3 * index + 2);
}

__device__ __forceinline__ void triangleFromTexture(int index, Vec3i & triangle) {
	triangle.x = tex1Dfetch(triangleTex, 3 * index);
	triangle.y = tex1Dfetch(triangleTex, 3 * index + 1);
	triangle.z = tex1Dfetch(triangleTex, 3 * index + 2);
}

__device__ __forceinline__ void verticesFromTexture(int index, Vec3f & v0, Vec3f & v1, Vec3f & v2) {
	vertexFromTexture(tex1Dfetch(triangleTex, 3 * index), v0);
	vertexFromTexture(tex1Dfetch(triangleTex, 3 * index + 1), v1);
	vertexFromTexture(tex1Dfetch(triangleTex, 3 * index + 2), v2);
}


}

#endif /* _CUDA_UTIL_H_ */
