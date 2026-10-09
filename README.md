**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* Chen Cheng
  * [LinkedIn](https://www.linkedin.com/in/chen-andrew-cheng-34a133229/), [GitHub](https://github.com/ischencheng)
* Tested on: Windows 11, i5-12500H @ 2.50GHz 16GB, NVIDIA GeForce RTX 2050 4GB (personal laptop)

### Screenshots

![boids](images/boids.gif)

This is 20,000 boids with the coherent grid and a fixed camera. The boids start at random positions in a cube and slowly gather into small flocks. The color shows the velocity of each boid.

![boids](images/boids.png)

### Naive

Boids is a simple model of how birds or fish move in a group. Every boid follows three rules: fly towards the center of its neighbors (cohesion), keep a small distance from them (separation), and match their velocity (alignment). In the naive version every boid checks every other boid, so one step is O(N²).

### Uniform Grid

A boid only cares about the boids within 5 units, the largest rule distance (I call it R). So I divide the space into cells and only check the cells near the boid. I label each boid with its cell, sort the boids by cell with `thrust::sort_by_key`, and then find where each cell starts and ends in the sorted array.

The cell width can be 2R or R. With 2R each boid checks 8 cells, and with R it checks 27 cells.

### Coherent Grid

In the uniform grid the indices of the boids in one cell are next to each other, but their positions and velocities are still all over the memory. In the coherent grid I also reorder the positions and velocities by cell, so the boids of one cell sit next to each other in memory and I don't need the extra `particleArrayIndices` lookup.

### Extra Credit: Grid-Looping

Instead of hard coding 8 or 27 cells, each boid computes the min and max cell index on each axis from `pos - R` and `pos + R` and loops over that range. A cell whose closest point is farther than R is skipped, because none of its boids can be a neighbor. This works for any cell width.

### Extra Credit: Shared Memory

The boids of one block are next to each other in the sorted order, so they need almost the same neighbor cells. In the shared memory version (`-mode shared`) the block goes over these cells one row (along x) at a time. For each row it finds the boids that any of its threads needs, loads them into shared memory together, and then every thread checks its own neighbors from there. To make a row of cells one range of boids, empty cells also need a start and end, which I get with `thrust::lower_bound` and `thrust::upper_bound`. I store each boid as a `float4` in shared memory, so a neighbor is one read.

### Performance Analysis

I added some command line options, so I don't have to recompile for every test:

```
cis5650_boids.exe -n 20000 -mode coherent -block 128 -cell 2 -novis -time 5
```

`-mode` is `naive`, `scattered`, `coherent` or `shared`, `-cell` is the cell width in units of R and `-novis` turns off drawing. With `-time 5` the program runs for 5 seconds after 1 second of warm up and prints the average fps. All numbers are from a Release build with v-sync off. They are a rough estimate: above 1000 fps they jump around by 10-20% between runs. Unless said otherwise the block size is 128 and the cell width is 2R.

![fps_vs_boids](images/fps_vs_boids.png)

This table shows the average fps for different numbers of boids ("vis" means with drawing).

| boids | naive | scattered | coherent | naive (vis) | scattered (vis) | coherent (vis) |
| --- | --- | --- | --- | --- | --- | --- |
| 1,000 | 2279.1 | 1742.3 | 1688.1 | 946.6 | 941.2 | 896.2 |
| 5,000 | 769.4 | 1778.7 | 1701.8 | 525.4 | 772.5 | 802.6 |
| 20,000 | 98.4 | 1102.0 | 1657.8 | 88.6 | 659.1 | 784.3 |
| 100,000 | 4.26 | 66.1 | 883.8 | 4.36 | 62.4 | 553.5 |
| 500,000 | - | 2.12 | 108.3 | - | 2.08 | 95.4 |
| 1,000,000 | - | 0.53 | 32.0 | - | 0.52 | 30.2 |

From the table, the coherent grid is the fastest for large flocks. At 100,000 boids it runs at 883.8 fps, vs 66.1 for scattered and 4.26 for naive, and it still gets 32 fps with 1,000,000 boids. For small flocks all three are about the same. This is because a frame is so short that the fixed cost of each frame (kernel launches, OpenGL, window events) is most of the time. Drawing caps every version at about 900 fps.

![fps_vs_blocksize](images/fps_vs_blocksize.png)

This is the fps for different block sizes with 50,000 boids and no drawing.

This table shows 8 cells (width 2R) vs 27 cells (width R), with no drawing.

| boids | scattered, 2R | scattered, R | coherent, 2R | coherent, R |
| --- | --- | --- | --- | --- |
| 20,000 | 1066.0 | 1330.3 | 1671.2 | 1688.0 |
| 100,000 | 66.2 | 102.9 | 882.8 | 1145.0 |
| 500,000 | 2.12 | 5.05 | 107.7 | 226.7 |

This table shows the grid-looping skip with width R. "Without skip" is the same code before I added the skip.

| version | boids | without skip | with skip |
| --- | --- | --- | --- |
| scattered | 100,000 | 82.6 | 102.9 |
| scattered | 500,000 | 3.84 | 5.05 |
| coherent | 100,000 | 1167.8 | 1145.0 |
| coherent | 500,000 | 234.5 | 226.7 |

Skipping the far cells makes the scattered grid 25-32% faster, because every boid it skips would have been a slow random read. For the coherent grid it doesn't really help (within about 3%), because those boids sit right next to boids it reads anyway. I also tried other cell widths on the coherent grid with 100,000 boids: 846.1 fps for 0.5R, 1118.6 for R, 809.3 for 1.5R, 866.0 for 2R and 524.5 for 3R, so R is the best width.

![fps_shared](images/fps_shared.png)

This table shows the coherent grid vs shared memory, with no drawing.

| boids | coherent, R | shared, R | coherent, 2R | shared, 2R |
| --- | --- | --- | --- | --- |
| 20,000 | 1879.3 | 1718.7 | 1852.6 | 1712.6 |
| 100,000 | 1066.6 | 1118.7 | 804.0 | 644.3 |
| 500,000 | 200.7 | 218.8 | 95.6 | 70.7 |
| 1,000,000 | 71.8 | 80.6 | 27.9 | 20.2 |

With width R, shared memory is faster from about 100,000 boids on, and 12% faster at 1,000,000 boids (13% with block size 256: 81.8 vs 72.3 fps). With width 2R it is 20-28% slower. This is because with width R all boids in a cell need exactly the same rows, so the block works on them together. With 2R each boid picks the 2 cells on its side on each axis, so the boids of one block need different rows, and the threads that don't need a row just wait at `__syncthreads()`. I expected a bigger gain. I think the coherent grid already gets most of its reads from the L1 cache, and on this GPU the L1 cache and shared memory are the same hardware.

### Answer to Questions

* **For each implementation, how does changing the number of boids affect performance? Why do you think this is?**

  All of them get slower with more boids. Naive is O(N²) because every boid checks every other boid: 5× more boids (20,000 → 100,000) makes it about 23× slower. The grids are much faster, but the box doesn't grow, so with more boids every cell also has more boids to check, and they slow down faster than O(N). The scattered grid drops the fastest, and I think this is because its random reads stop fitting in the cache.

* **For each implementation, how does changing the block count and block size affect performance? Why do you think this is?**

  Roughly speaking, for naive and coherent the block size doesn't matter much from 64 to 1024 (within 15%), and 32 is the slowest. I think this is because one SM can only hold 16 blocks on this GPU, so 32-thread blocks give only 16 warps per SM, which is not enough to hide the memory latency. The scattered grid is the odd one: it is fastest with 32 (448.9 fps vs about 330). My guess is that with fewer warps, fewer random reads compete for the small L1 cache.

* **For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?**

  Yes, a lot: it is 13× faster than scattered at 100,000 boids and about 60× faster at 1,000,000. I expected it to be faster, but not by this much. I think the main reason is that the threads of a warp are now boids in the same or nearby cells, so they read almost the same memory at the same time and share it through the cache. In the scattered version the threads of a warp are boids from random places, so every thread reads different memory.

* **Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!**

  Yes, 27 cells with width R was faster, up to 2.4× at 500,000 boids. What matters is the volume I search: 8 cells of width 2R make a (4R)³ = 64R³ box, but 27 cells of width R make only a (3R)³ = 27R³ box. So each boid checks about 2.4× fewer boids. For small flocks the two are about the same, because there are few boids to check anyway, and the smaller cells mean more cells to visit and a bigger grid to reset.
