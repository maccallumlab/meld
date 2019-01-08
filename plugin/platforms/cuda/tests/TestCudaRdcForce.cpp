/*
   Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
   All rights reserved
*/


#include "RdcForce.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/Context.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>

using namespace MeldPlugin;
using namespace OpenMM;
using namespace std;

extern "C" OPENMM_EXPORT void registerMeldCudaKernelFactories();


void testTranslationInvariance() {
    // Setup RNG
    default_random_engine generator;
    generator.seed(1979);
    normal_distribution<double> distribution(0.0, 1.0);

    System system;

    // Add the particles with random positions
    const int numParticles = 20;
    const int numRdc = 10;
    vector<Vec3> positions;
    for(int i=0; i<numParticles; i++) {
        float x = 0.1 * distribution(generator);
        float y = 0.1 * distribution(generator);
        float z = 0.1 * distribution(generator);
        positions.push_back(Vec3(x, y, z));
        system.addParticle(1.0);
    }

    RdcForce* force = new RdcForce();
    float weight = 1.0;
    float fc = 2000.0;
    float tol = 0.0;
    float kappa = 10000.0;
    vector<int> rest_ids;

    for(int i=0; i<numRdc; i++) {
        float obs = 5.0 * distribution(generator);
        int restIdx = force->addRdcRestraint(
            2*i, 2*i+1,
            kappa,
            obs,
            tol,
            fc,
            weight
        );
        rest_ids.push_back(restIdx);
    }

    force->addExperiment(rest_ids);
    system.addForce(force);

    vector<Vec3> pos1(positions);
    vector<Vec3> pos2(positions);

    const float pert = 42.0;
    for(int i=0; i<numParticles; i++) {
        pos2[i][0] += pert;
    }

    VerletIntegrator integ(1.0);
    Platform& platform = Platform::getPlatformByName("CUDA");
    Context context(system, integ, platform);

    // compute the initial energy and forces
    context.setPositions(pos1);
    State state1 = context.getState(State::Energy | State::Forces);
    double energy1 = state1.getPotentialEnergy();
    vector<Vec3> forces1 = state1.getForces();

    // compute the energy and forces for the translated system
    context.setPositions(pos2);
    State state2 = context.getState(State::Energy | State::Forces);
    double energy2 = state2.getPotentialEnergy();
    vector<Vec3> forces2 = state2.getForces();

    // The energy should match
    ASSERT_EQUAL_TOL(energy1, energy2, 1e-4);

    // The forces should match
    const float delta = 1e-4;
    for(int i=0; i<numParticles; i++) {
        for(int j=0; j<3; j++) {
            ASSERT_EQUAL_TOL(forces1[i][j], forces2[i][j], 1e-3);
        }
    }
}


void testRotationInvariance() {
    // Setup RNG
    default_random_engine generator;
    generator.seed(1979);
    normal_distribution<double> distribution(0.0, 1.0);

    System system;

    // Add the particles with random positions
    const int numParticles = 20;
    const int numRdc = 10;
    vector<Vec3> positions;
    for(int i=0; i<numParticles; i++) {
        float x = 0.1 * distribution(generator);
        float y = 0.1 * distribution(generator);
        float z = 0.1 * distribution(generator);
        positions.push_back(Vec3(x, y, z));
        system.addParticle(1.0);
    }

    RdcForce* force = new RdcForce();
    float weight = 1.0;
    float fc = 2000.0;
    float tol = 0.0;
    float kappa = 10000.0;
    vector<int> rest_ids;

    for(int i=0; i<numRdc; i++) {
        float obs = 5.0 * distribution(generator);
        int restIdx = force->addRdcRestraint(
            2*i, 2*i+1,
            kappa,
            obs,
            tol,
            fc,
            weight
        );
        rest_ids.push_back(restIdx);
    }

    force->addExperiment(rest_ids);
    system.addForce(force);

    vector<Vec3> pos1(positions);
    vector<Vec3> pos2(positions);

    // we'll rotate around the z-axis by 180 degrees
    // so x becomes -x and y becomes -y
    for(int i=0; i<numParticles; i++) {
        pos2[i][0] = -pos1[i][0];
        pos2[i][1] = -pos1[i][1];
        pos2[i][2] = pos1[i][2];
    }

    VerletIntegrator integ(1.0);
    Platform& platform = Platform::getPlatformByName("CUDA");
    Context context(system, integ, platform);

    // compute the initial energy and forces
    context.setPositions(pos1);
    State state1 = context.getState(State::Energy | State::Forces);
    double energy1 = state1.getPotentialEnergy();
    vector<Vec3> forces1 = state1.getForces();

    // compute the energy and forces for the translated system
    context.setPositions(pos2);
    State state2 = context.getState(State::Energy | State::Forces);
    double energy2 = state2.getPotentialEnergy();
    vector<Vec3> forces2 = state2.getForces();

    // The energy should match
    ASSERT_EQUAL_TOL(energy1, energy2, 1e-4);

    // The forces should match after accounting for rotation
    const float delta = 1e-4;
    for(int i=0; i<numParticles; i++) {
        ASSERT_EQUAL_TOL(-forces1[i][0], forces2[i][0], 1e-3);
        ASSERT_EQUAL_TOL(-forces1[i][1], forces2[i][1], 1e-3);
        ASSERT_EQUAL_TOL(forces1[i][2], forces2[i][2], 1e-3);
    }
}


void testForceMatchesFiniteDifference() {
    // Setup RNG
    default_random_engine generator;
    normal_distribution<double> distribution(0.0, 1.0);
    generator.seed(1979);

    System system;

    // Add the particles with random positions
    const int numParticles = 20;
    const int numRdc = 10;
    vector<Vec3> positions;
    for(int i=0; i<numParticles; i++) {
        float x = 0.1 * distribution(generator);
        float y = 0.1 * distribution(generator);
        float z = 0.1 * distribution(generator);
        positions.push_back(Vec3(x, y, z));
        system.addParticle(1.0);
    }

    RdcForce* force = new RdcForce();
    float weight = 1.0;
    float fc = 2000.0;
    float tol = 0.0;
    float kappa = 10000.0;
    vector<int> rest_ids;

    for(int i=0; i<numRdc; i++) {
        float obs = 5.0 * distribution(generator);
        int restIdx = force->addRdcRestraint(
            2*i, 2*i+1,
            kappa,
            obs,
            tol,
            fc,
            weight
        );
        rest_ids.push_back(restIdx);
    }

    force->addExperiment(rest_ids);
    system.addForce(force);

    // Compute the forces and energy.
    VerletIntegrator integ(1.0);
    Platform& platform = Platform::getPlatformByName("CUDA");
    Context context(system, integ, platform);
    context.setPositions(positions);
    State state1 = context.getState(State::Energy | State::Forces);
    double energy1 = state1.getPotentialEnergy();
    vector<Vec3> forces = state1.getForces();

    // loop over all particles and over all 3 directions
    const float delta = 1e-3;
    for(int i=0; i<numParticles; i++) {
        for(int j=0; j<3; j++) {
            vector<Vec3> pos1(positions);
            vector<Vec3> pos2(positions);
            pos1[i][j] += delta;
            pos2[i][j] -= delta;

            context.setPositions(pos1);
            double energy2 = context.getState(State::Energy).getPotentialEnergy();
            context.setPositions(pos2);
            double energy3 = context.getState(State::Energy).getPotentialEnergy();

            float fd_force = (energy3 - energy2) / (2 * delta);
            float actual_force = forces[i][j];

            // See if the forces match what is expected.
            // The tolerance is coarse, as chaning the
            // coordinates will change the alignment,
            // which is not included in the force calculation.
            ASSERT_EQUAL_TOL(fd_force, actual_force, 1e-1);
        }
    }
}


int main(int argc, char* argv[]) {
    try {
        registerMeldCudaKernelFactories();
        if (argc > 1)
            Platform::getPlatformByName("CUDA").setPropertyDefaultValue("CudaPrecision", string(argv[1]));
        testTranslationInvariance();
        testRotationInvariance();
        testForceMatchesFiniteDifference();
    }
    catch(const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}
