/**
* \file		BVHUtil.h
* \author	Daniel Meister
* \date		2017/01/23
* \brief	A header file containing useful functions.
*/

#ifndef _BVH_UTIL_H_
#define _BVH_UTIL_H_

#define LOG_WARP_THREADS 5
#define WARP_THREADS (1 << LOG_WARP_THREADS)

#define divCeil(a, b) (((a) + (b) - 1) / (b))
#define divCeilLog(a, b) (((a) + (1 << (b)) - 1) >> (b))

#endif /* _BVH_UTIL_H_ */
