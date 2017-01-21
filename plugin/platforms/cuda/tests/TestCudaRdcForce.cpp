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
#include <vector>

using namespace MeldPlugin;
using namespace OpenMM;
using namespace std;

extern "C" OPENMM_EXPORT void registerMeldCudaKernelFactories();

void testRdcForce() {
    const int numParticles = 2;
    System system;
    vector<Vec3> positions(numParticles);
    system.addParticle(1.0);
    positions[0] = Vec3(0.0, 0.0, 0.0);
    system.addParticle(1.0);
    positions[1] = Vec3(2.5, 0.0, 0.0);

    RdcForce* force = new RdcForce();
    float weight = 1.0;
    int restIdx = force->addRdcRestraint(0, 1, 10.0, 0.0, 10.0, 25000.0, weight);
    vector<int> rest_ids(1);
    rest_ids[0] = restIdx;
    force->addExperiment(rest_ids);
    system.addForce(force);

    // Compute the forces and energy.
    VerletIntegrator integ(1.0);
    Platform& platform = Platform::getPlatformByName("CUDA");
    Context context(system, integ, platform);
    context.setPositions(positions);
    State state = context.getState(State::Energy | State::Forces);
    
    // See if the energy is correct.
    double expectedEnergy = state.getPotentialEnergy();
    ASSERT_EQUAL_TOL(expectedEnergy, state.getPotentialEnergy(), 1e-5);

}

void testChangingParameters() {
    const int numParticles = 2;
    System system;
    vector<Vec3> positions(numParticles);
    system.addParticle(1.0);
    positions[0] = Vec3(0.0, 0.0, 0.0);
    system.addParticle(1.0);
    positions[1] = Vec3(3.5, 0.0, 0.0);

    RdcForce* force = new RdcForce();
    float weight = 1.0;
    int restIdx = force->addRdcRestraint(0, 1, 10.0, 0.0, 10.0, 25000.0, weight);
    vector<int> rest_ids(1);
    rest_ids[0] = restIdx;
    force->addExperiment(rest_ids);
    system.addForce(force);

    // Compute the forces and energy.
    VerletIntegrator integ(1.0);
    Platform& platform = Platform::getPlatformByName("CUDA");
    Context context(system, integ, platform);
    context.setPositions(positions);
    State state = context.getState(State::Energy | State::Forces);

    // See if the energy is correct.
    double expectedEnergy = state.getPotentialEnergy();
    ASSERT_EQUAL_TOL(expectedEnergy, state.getPotentialEnergy(), 1e-5);

    // Modify the parameters.
    float weight2 = 0.1;
    force->updateRdcRestraint(restIdx, 0, 1, 10.0, 0.0, 10.0, 25000.0, weight2);
    force->updateParametersInContext(context);
    state = context.getState(State::Energy);

    // See if the energy is correct after modifying force const.
    expectedEnergy = state.getPotentialEnergy();
    ASSERT_EQUAL_TOL(expectedEnergy, state.getPotentialEnergy(), 1e-5);
}

void testTranslatingSystemDoesNotChangeEnergy() {
    const int numParticles = 2;
    System system;
    vector<Vec3> positions(numParticles);
    system.addParticle(1.0);
    positions[0] = Vec3(0.0, 0.0, 0.0);
    system.addParticle(1.0);
    positions[1] = Vec3(3.5, 0.0, 0.0);

    RdcForce* force = new RdcForce();
    float weight = 1.0;
    int restIdx = force->addRdcRestraint(0, 1, 10.0, 0.0, 10.0, 25000.0, weight);
    vector<int> rest_ids(1);
    rest_ids[0] = restIdx;
    force->addExperiment(rest_ids);
    system.addForce(force);

    // Compute the forces and energy.
    VerletIntegrator integ(1.0);
    Platform& platform = Platform::getPlatformByName("CUDA");
    Context context(system, integ, platform);
    context.setPositions(positions);
    State state1 = context.getState(State::Energy | State::Forces);
    double energy1 = state1.getPotentialEnergy();

    // Translate particles
    positions[0][0] += 10.0;
    positions[1][0] += 10.0;
    context.setPositions(positions);
    State state2 = context.getState(State::Energy | State::Forces);
    double energy2 = state2.getPotentialEnergy();

    // See if the energy is correct.
    ASSERT_EQUAL_TOL(energy1, energy2, 1e-5);
}

void testRotatingSystemDoesNotChangeEnergy() {
    const int numParticles = 2;
    System system;
    vector<Vec3> positions(numParticles);
    system.addParticle(1.0);
    positions[0] = Vec3(0.0, 0.0, 0.0);
    system.addParticle(1.0);
    positions[1] = Vec3(3.5, 0.0, 0.0);

    RdcForce* force = new RdcForce();
    float weight = 1.0;
    int restIdx = force->addRdcRestraint(0, 1, 10.0, 0.0, 10.0, 25000.0, weight);
    vector<int> rest_ids(1);
    rest_ids[0] = restIdx;
    force->addExperiment(rest_ids);
    system.addForce(force);

    // Compute the forces and energy.
    VerletIntegrator integ(1.0);
    Platform& platform = Platform::getPlatformByName("CUDA");
    Context context(system, integ, platform);
    context.setPositions(positions);
    State state1 = context.getState(State::Energy | State::Forces);
    double energy1 = state1.getPotentialEnergy();

    // 90 degree rotation
    positions[1][0] = 0.0;
    positions[1][1] = 3.5;
    context.setPositions(positions);
    State state2 = context.getState(State::Energy | State::Forces);
    double energy2 = state2.getPotentialEnergy();

    // See if the energy is correct.
    ASSERT_EQUAL_TOL(energy1, energy2, 1e-5);
}

int main(int argc, char* argv[]) {
    try {
        registerMeldCudaKernelFactories();
        if (argc > 1)
            Platform::getPlatformByName("CUDA").setPropertyDefaultValue("CudaPrecision", string(argv[1]));
        testRdcForce();
        testChangingParameters();
        testTranslatingSystemDoesNotChangeEnergy();
        testRotatingSystemDoesNotChangeEnergy();
    }
    catch(const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}
