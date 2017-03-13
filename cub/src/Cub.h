/**
* \file		Cub.h
* \author	Daniel Meister
* \date		2017/01/23
* \brief	Cub wrapper ehader file.
*/

#ifndef _CUB_H_
#define _CUB_H_

namespace Cub {

float sort(
	int numberOfItems,
	unsigned int * keys0,
	unsigned int * keys1,
	int * values0,
	int * values1,
	bool & swapBuffers
);

float sort(
	int numberOfItems,
	unsigned long long * keys0,
	unsigned long long * keys1,
	int * values0,
	int * values1,
	bool & swapBuffers
);

};

#endif /* _CUB_H_ */
