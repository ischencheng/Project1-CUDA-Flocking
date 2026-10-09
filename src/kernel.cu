#define GLM_FORCE_CUDA

#include <cuda.h>
#include "kernel.h"
#include "utilityCore.hpp"

#include <climits>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>

#include <glm/glm.hpp>

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. Set in Boids::initSimulation. */
int blockSize = 128;

// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

// Grid cell width in units of the max rule distance. Set in Boids::initSimulation.
// 2 -> each boid checks 8 cells, 1 -> each boid checks 27 cells
float cellWidthScale = 2.0f;

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.
// Positions are reshuffled into dev_pos2. Velocities are reshuffled into
// dev_vel2, because the unsorted vel1 is not needed after that.
glm::vec3 *dev_pos2;

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N, int blockSize, float cellWidthScale) {
  numObjects = N;
  ::blockSize = blockSize;
  ::cellWidthScale = cellWidthScale;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  // Initialize velocity to 0
  cudaMemset(dev_vel1, 0, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMemset dev_vel1 failed!");

  cudaMemset(dev_vel2, 0, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMemset dev_vel2 failed!");

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // LOOK-2.1 computing grid params
  gridCellWidth = cellWidthScale * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.
  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

  dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
  dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);

  cudaMalloc((void**)&dev_pos2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos2 failed!");

  cudaDeviceSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

// Sums of the three rules over all neighbors of one boid
struct RuleSums {
  glm::vec3 center = glm::vec3(0.0f);
  glm::vec3 separate = glm::vec3(0.0f);
  glm::vec3 perceivedVel = glm::vec3(0.0f);
  int count1 = 0;
  int count3 = 0;
};

// Add another boid (not itself) to the rule sums if it is close enough
__device__ void addNeighbor(RuleSums &sums, glm::vec3 selfPos,
  glm::vec3 otherPos, glm::vec3 otherVel) {
  float dist = glm::distance(otherPos, selfPos);
  // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
  if (dist < rule1Distance) {
    sums.center += otherPos;
    sums.count1++;
  }
  // Rule 2: boids try to stay a distance d away from each other
  if (dist < rule2Distance) {
    sums.separate -= otherPos - selfPos;
  }
  // Rule 3: boids try to match the speed of surrounding boids
  if (dist < rule3Distance) {
    sums.perceivedVel += otherVel;
    sums.count3++;
  }
}

__device__ glm::vec3 rulesToVelocityChange(const RuleSums &sums, glm::vec3 selfPos) {
  glm::vec3 dv(0.0f);
  if (sums.count1 > 0) {
    dv += (sums.center / (float)sums.count1 - selfPos) * rule1Scale;
  }
  dv += sums.separate * rule2Scale;
  if (sums.count3 > 0) {
    dv += sums.perceivedVel / (float)sums.count3 * rule3Scale;
  }
  return dv;
}

__device__ glm::vec3 clampSpeed(glm::vec3 vel) {
  float speed = glm::length(vel);
  return speed > maxSpeed ? vel * (maxSpeed / speed) : vel;
}

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
  RuleSums sums;
  for (int i = 0; i < N; i++) {
    if (i != iSelf) {
      addNeighbor(sums, pos[iSelf], pos[i], vel[i]);
    }
  }
  return rulesToVelocityChange(sums, pos[iSelf]);
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  // Compute a new velocity based on pos and vel1
  glm::vec3 newVel = vel1[index] + computeVelocityChange(N, index, pos, vel1);
  // Clamp the speed
  // Record the new velocity into vel2. Question: why NOT vel1?
  // Other threads may still be reading vel1 of this boid for their own update.
  vel2[index] = clampSpeed(newVel);
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

// Range of cells (inclusive) that overlap the box of size 2 * maxDistance
// around the boid. With cell width = 2 * maxDistance this is 2x2x2 = 8 cells,
// with cell width = maxDistance it is 3x3x3 = 27 cells.
__device__ void findNeighborCells(glm::vec3 pos, glm::vec3 gridMin,
  float inverseCellWidth, int gridResolution, glm::ivec3 &minCell, glm::ivec3 &maxCell) {
  float maxDistance = imax(imax(rule1Distance, rule2Distance), rule3Distance);
  minCell = glm::ivec3(glm::floor((pos - gridMin - maxDistance) * inverseCellWidth));
  maxCell = glm::ivec3(glm::floor((pos - gridMin + maxDistance) * inverseCellWidth));
  minCell = glm::clamp(minCell, 0, gridResolution - 1);
  maxCell = glm::clamp(maxCell, 0, gridResolution - 1);
}

// Grid-looping: does any part of cell (x, y, z) lie within maxDistance of the
// boid? The corner cells of the range above often don't, so we can skip them.
__device__ bool cellTouchesNeighborhood(glm::vec3 pos, glm::vec3 gridMin,
  float cellWidth, int x, int y, int z) {
  float maxDistance = imax(imax(rule1Distance, rule2Distance), rule3Distance);
  glm::vec3 cellMin = gridMin + glm::vec3(x, y, z) * cellWidth;
  glm::vec3 closest = glm::clamp(pos, cellMin, cellMin + cellWidth);
  return glm::distance(closest, pos) <= maxDistance;
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
    // TODO-2.1
    // - Label each boid with the index of its grid cell.
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
      return;
    }
    glm::ivec3 cell = glm::ivec3(glm::floor((pos[index] - gridMin) * inverseCellWidth));
    gridIndices[index] = gridIndex3Dto1D(cell.x, cell.y, cell.z, gridResolution);
    indices[index] = index;
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {
  // TODO-2.1
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  int cell = particleGridIndices[index];
  if (index == 0 || particleGridIndices[index - 1] != cell) {
    gridCellStartIndices[cell] = index;
  }
  // end is exclusive, so the boids of a cell are [start, end)
  if (index == N - 1 || particleGridIndices[index + 1] != cell) {
    gridCellEndIndices[cell] = index + 1;
  }
}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  glm::vec3 selfPos = pos[index];
  glm::ivec3 minCell, maxCell;
  findNeighborCells(selfPos, gridMin, inverseCellWidth, gridResolution, minCell, maxCell);

  RuleSums sums;
  for (int z = minCell.z; z <= maxCell.z; z++) {
    for (int y = minCell.y; y <= maxCell.y; y++) {
      for (int x = minCell.x; x <= maxCell.x; x++) {
        if (!cellTouchesNeighborhood(selfPos, gridMin, cellWidth, x, y, z)) {
          continue;
        }
        int cell = gridIndex3Dto1D(x, y, z, gridResolution);
        // empty cells have start = end = -1, so the loop is skipped
        for (int i = gridCellStartIndices[cell]; i < gridCellEndIndices[cell]; i++) {
          int other = particleArrayIndices[i];
          if (other != index) {
            addNeighbor(sums, selfPos, pos[other], vel1[other]);
          }
        }
      }
    }
  }
  vel2[index] = clampSpeed(vel1[index] + rulesToVelocityChange(sums, selfPos));
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
  // except with one less level of indirection.
  // This should expect gridCellStartIndices and gridCellEndIndices to refer
  // directly to pos and vel1.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  //   DIFFERENCE: For best results, consider what order the cells should be
  //   checked in to maximize the memory benefits of reordering the boids data.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  glm::vec3 selfPos = pos[index];
  glm::ivec3 minCell, maxCell;
  findNeighborCells(selfPos, gridMin, inverseCellWidth, gridResolution, minCell, maxCell);

  RuleSums sums;
  // x changes fastest in gridIndex3Dto1D, so x is the inner loop. Then the
  // cells we read one after another are also next to each other in memory.
  for (int z = minCell.z; z <= maxCell.z; z++) {
    for (int y = minCell.y; y <= maxCell.y; y++) {
      for (int x = minCell.x; x <= maxCell.x; x++) {
        if (!cellTouchesNeighborhood(selfPos, gridMin, cellWidth, x, y, z)) {
          continue;
        }
        int cell = gridIndex3Dto1D(x, y, z, gridResolution);
        for (int i = gridCellStartIndices[cell]; i < gridCellEndIndices[cell]; i++) {
          if (i != index) {
            addNeighbor(sums, selfPos, pos[i], vel1[i]);
          }
        }
      }
    }
  }
  vel2[index] = clampSpeed(vel1[index] + rulesToVelocityChange(sums, selfPos));
}

// Gather pos and vel into the sorted cell order
__global__ void kernReshuffle(int N, int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel, glm::vec3 *posSorted, glm::vec3 *velSorted) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  int src = particleArrayIndices[index];
  posSorted[index] = pos[src];
  velSorted[index] = vel[src];
}

// Extra credit: shared memory. Works on the coherent data like
// kernUpdateVelNeighborSearchCoherent. The boids of one block are next to each
// other in the sorted order, so they need almost the same neighbor cells. The
// block goes over its neighbor cells one row (along x) at a time, loads the
// boids of the row into shared memory together, and every thread reads its own
// neighbors from there. Every cell needs a [start, end) range here, also the
// empty ones, so that a row of cells is one range of boids.
__global__ void kernUpdateVelNeighborSearchShared(
  int N, int gridResolution, glm::vec3 gridMin, float inverseCellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // blockDim.x positions, then blockDim.x velocities. float4 instead of vec3,
  // so one boid is a single 16 byte read.
  extern __shared__ float4 tile[];
  float4 *tilePos = tile;
  float4 *tileVel = tile + blockDim.x;
  __shared__ int blockMin[3], blockMax[3], rowStart, rowEnd;

  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  // threads past N have no boid, but they still help to load the tiles
  bool active = index < N;
  glm::vec3 selfPos(0.0f);
  glm::ivec3 minCell, maxCell;
  if (active) {
    selfPos = pos[index];
    findNeighborCells(selfPos, gridMin, inverseCellWidth, gridResolution, minCell, maxCell);
  }

  // box of cells that covers the neighbor cells of all boids in the block
  if (threadIdx.x == 0) {
    for (int k = 0; k < 3; k++) {
      blockMin[k] = INT_MAX;
      blockMax[k] = -1;
    }
  }
  __syncthreads();
  if (active) {
    for (int k = 0; k < 3; k++) {
      atomicMin(&blockMin[k], minCell[k]);
      atomicMax(&blockMax[k], maxCell[k]);
    }
  }
  __syncthreads();

  RuleSums sums;
  for (int z = blockMin[2]; z <= blockMax[2]; z++) {
    for (int y = blockMin[1]; y <= blockMax[1]; y++) {
      // the boids this thread needs from this row
      int myStart = INT_MAX, myEnd = -1;
      if (active && y >= minCell.y && y <= maxCell.y && z >= minCell.z && z <= maxCell.z) {
        myStart = gridCellStartIndices[gridIndex3Dto1D(minCell.x, y, z, gridResolution)];
        myEnd = gridCellEndIndices[gridIndex3Dto1D(maxCell.x, y, z, gridResolution)];
      }
      // the part of the row that any thread of the block needs
      if (threadIdx.x == 0) {
        rowStart = INT_MAX;
        rowEnd = -1;
      }
      __syncthreads();
      if (myStart < myEnd) {
        atomicMin(&rowStart, myStart);
        atomicMax(&rowEnd, myEnd);
      }
      __syncthreads();
      int start = rowStart, end = rowEnd;
      if (start >= end) {
        // nobody needs this row. Still sync, so thread 0 doesn't reset rowStart
        // for the next row while other threads are reading it.
        __syncthreads();
        continue;
      }

      for (int tileStart = start; tileStart < end; tileStart += blockDim.x) {
        int j = tileStart + threadIdx.x;
        if (j < end) {
          glm::vec3 p = pos[j], v = vel1[j];
          tilePos[threadIdx.x] = make_float4(p.x, p.y, p.z, 0.0f);
          tileVel[threadIdx.x] = make_float4(v.x, v.y, v.z, 0.0f);
        }
        __syncthreads();
        int from = imax(myStart, tileStart);
        int to = imin(myEnd, tileStart + (int)blockDim.x);
        for (int i = from; i < to; i++) {
          if (i != index) {
            float4 p = tilePos[i - tileStart], v = tileVel[i - tileStart];
            addNeighbor(sums, selfPos, glm::vec3(p.x, p.y, p.z), glm::vec3(v.x, v.y, v.z));
          }
        }
        __syncthreads();
      }
    }
  }
  if (active) {
    vel2[index] = clampSpeed(vel1[index] + rulesToVelocityChange(sums, selfPos));
  }
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
  kernUpdateVelocityBruteForce<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_pos, dev_vel1, dev_vel2);
  checkCUDAErrorWithLine("kernUpdateVelocityBruteForce failed!");

  kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel2);
  checkCUDAErrorWithLine("kernUpdatePos failed!");

  // TODO-1.2 ping-pong the velocity buffers
  std::swap(dev_vel1, dev_vel2);
}

// Label the boids with their cells, sort them by cell and find where each
// cell starts and ends in the sorted array. With fillEmptyCells, an empty cell
// gets start = end = the place where its boids would be, instead of -1.
void buildUniformGrid(bool fillEmptyCells = false) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  dim3 cellBlocksPerGrid((gridCellCount + blockSize - 1) / blockSize);

  kernComputeIndices<<<fullBlocksPerGrid, blockSize>>>(numObjects, gridSideCount,
    gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
  checkCUDAErrorWithLine("kernComputeIndices failed!");

  thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects,
    dev_thrust_particleArrayIndices);

  if (fillEmptyCells) {
    // binary search every cell index in the sorted cell indices
    thrust::counting_iterator<int> cells(0);
    thrust::lower_bound(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects,
      cells, cells + gridCellCount, thrust::device_ptr<int>(dev_gridCellStartIndices));
    thrust::upper_bound(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects,
      cells, cells + gridCellCount, thrust::device_ptr<int>(dev_gridCellEndIndices));
    return;
  }

  // -1 marks an empty cell
  kernResetIntBuffer<<<cellBlocksPerGrid, blockSize>>>(gridCellCount, dev_gridCellStartIndices, -1);
  kernResetIntBuffer<<<cellBlocksPerGrid, blockSize>>>(gridCellCount, dev_gridCellEndIndices, -1);
  checkCUDAErrorWithLine("kernResetIntBuffer failed!");

  kernIdentifyCellStartEnd<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_particleGridIndices,
    dev_gridCellStartIndices, dev_gridCellEndIndices);
  checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");
}

void Boids::stepSimulationScatteredGrid(float dt) {
  // TODO-2.1
  // Uniform Grid Neighbor search using Thrust sort.
  // In Parallel:
  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  buildUniformGrid();

  // - Perform velocity updates using neighbor search
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  kernUpdateVelNeighborSearchScattered<<<fullBlocksPerGrid, blockSize>>>(numObjects, gridSideCount,
    gridMinimum, gridInverseCellWidth, gridCellWidth,
    dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices,
    dev_pos, dev_vel1, dev_vel2);
  checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered failed!");

  // - Update positions
  kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel2);
  checkCUDAErrorWithLine("kernUpdatePos failed!");

  // - Ping-pong buffers as needed
  std::swap(dev_vel1, dev_vel2);
}

void Boids::stepSimulationCoherentGrid(float dt) {
  // TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
  // In Parallel:
  // - Label each particle with its array index as well as its grid index.
  //   Use 2x width grids
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  buildUniformGrid();

  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
  //   the particle data in the simulation array.
  //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  kernReshuffle<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_particleArrayIndices,
    dev_pos, dev_vel1, dev_pos2, dev_vel2);
  checkCUDAErrorWithLine("kernReshuffle failed!");

  // - Perform velocity updates using neighbor search
  // sorted velocities are in vel2 now, so the new velocities go into vel1
  kernUpdateVelNeighborSearchCoherent<<<fullBlocksPerGrid, blockSize>>>(numObjects, gridSideCount,
    gridMinimum, gridInverseCellWidth, gridCellWidth,
    dev_gridCellStartIndices, dev_gridCellEndIndices,
    dev_pos2, dev_vel2, dev_vel1);
  checkCUDAErrorWithLine("kernUpdateVelNeighborSearchCoherent failed!");

  // - Update positions
  kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos2, dev_vel1);
  checkCUDAErrorWithLine("kernUpdatePos failed!");

  // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.
  // vel1 already has the new velocities, only the positions need to swap.
  // The boids just stay in the sorted order for the next step.
  std::swap(dev_pos, dev_pos2);
}

// Same as the coherent grid, except for the cell ranges and the velocity kernel
void Boids::stepSimulationSharedGrid(float dt) {
  buildUniformGrid(true);

  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  kernReshuffle<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_particleArrayIndices,
    dev_pos, dev_vel1, dev_pos2, dev_vel2);
  checkCUDAErrorWithLine("kernReshuffle failed!");

  // shared memory for blockSize positions and blockSize velocities
  int sharedBytes = 2 * blockSize * sizeof(float4);
  kernUpdateVelNeighborSearchShared<<<fullBlocksPerGrid, blockSize, sharedBytes>>>(numObjects,
    gridSideCount, gridMinimum, gridInverseCellWidth,
    dev_gridCellStartIndices, dev_gridCellEndIndices,
    dev_pos2, dev_vel2, dev_vel1);
  checkCUDAErrorWithLine("kernUpdateVelNeighborSearchShared failed!");

  kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos2, dev_vel1);
  checkCUDAErrorWithLine("kernUpdatePos failed!");

  std::swap(dev_pos, dev_pos2);
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  // TODO-2.1 TODO-2.3 - Free any additional buffers here.
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);
  cudaFree(dev_pos2);
}

void Boids::unitTest() {
  // LOOK-1.2 Feel free to write additional tests here.

  // test unstable sort
  int *dev_intKeys;
  int *dev_intValues;
  int N = 10;

  std::unique_ptr<int[]>intKeys{ new int[N] };
  std::unique_ptr<int[]>intValues{ new int[N] };

  intKeys[0] = 0; intValues[0] = 0;
  intKeys[1] = 1; intValues[1] = 1;
  intKeys[2] = 0; intValues[2] = 2;
  intKeys[3] = 3; intValues[3] = 3;
  intKeys[4] = 0; intValues[4] = 4;
  intKeys[5] = 2; intValues[5] = 5;
  intKeys[6] = 2; intValues[6] = 6;
  intKeys[7] = 0; intValues[7] = 7;
  intKeys[8] = 5; intValues[8] = 8;
  intKeys[9] = 6; intValues[9] = 9;

  cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  // LOOK-2.1 Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  std::cout << "after unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}
