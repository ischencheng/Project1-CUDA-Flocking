#pragma once

namespace Boids {
    void initSimulation(int N, int blockSize = 128, float cellWidthScale = 2.0f);
    void stepSimulationNaive(float dt);
    void stepSimulationScatteredGrid(float dt);
    void stepSimulationCoherentGrid(float dt);
    void stepSimulationSharedGrid(float dt);
    void copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities);

    void endSimulation();
    void unitTest();
}
