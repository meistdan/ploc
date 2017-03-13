/**
* \file		Cub.cu
* \author	Daniel Meister
* \date		2017/01/23
* \brief	Cub wrapper source file.
*/

#include "Cub.h"
#include <cub/cub.cuh>
#include <cuda_runtime_api.h>

namespace Cub {

template <typename T>
float sort(
	int numberOfItems,
	T * keys0,
	T * keys1,
	int * values0,
	int * values1,
	bool & swapBuffers
	) {

	cub::DoubleBuffer<T> keysBuffer(keys0, keys1);
	cub::DoubleBuffer<int> valuesBuffer(values0, values1);

	void * tempStorage = nullptr;
	size_t storageSize = 0;
	cub::DeviceRadixSort::SortPairs(tempStorage, storageSize, keysBuffer, valuesBuffer, numberOfItems);
	cudaMalloc(&tempStorage, storageSize);

	float elapsedTime = 0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	cub::DeviceRadixSort::SortPairs(tempStorage, storageSize, keysBuffer, valuesBuffer, numberOfItems);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaFree(tempStorage);

	swapBuffers = keysBuffer.selector != 0;

	return elapsedTime * 1.0e-3f;

}

float sort(
	int numberOfItems,
	unsigned int * keys0,
	unsigned int * keys1,
	int * values0,
	int * values1,
	bool & swapBuffers
	) {
	return sort<unsigned int>(numberOfItems, keys0, keys1, values0, values1, swapBuffers);
}

float sort(
	int numberOfItems,
	unsigned long long * keys0,
	unsigned long long * keys1,
	int * values0,
	int * values1,
	bool & swapBuffers
	) {
	return sort<unsigned long long int>(numberOfItems, keys0, keys1, values0, values1, swapBuffers);
}

};