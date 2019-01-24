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
#include "openmm/LocalEnergyMinimizer.h"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>

using namespace MeldPlugin;
using namespace OpenMM;
using namespace std;

extern "C" OPENMM_EXPORT void registerMeldCudaKernelFactories();


void testUpdateForceConstants(float cut) {
    // Setup RNG
    default_random_engine generator;
    generator.seed(1979);
    normal_distribution<double> distribution(0.0, 1.0);

    System system;

    // Add the particles with random positions
    const int numParticles = 256;
    const int numRdc = 128;
    vector<Vec3> positions;
    for(int i=0; i<numParticles; i++) {
        float x = 0.1 * distribution(generator);
        float y = 0.1 * distribution(generator);
        float z = 0.1 * distribution(generator);
        positions.push_back(Vec3(x, y, z));
        system.addParticle(1.0);
    }


    float weight = 1.0;
    float tol = 0.0;
    float kappa = 10000.0;
    float fc1 = 10.0;
    float fc2 = 20.0;
    float fc0 = 0.0;

    RdcForce* force = new RdcForce();
    vector<int> rest_ids;

    for(int i=0; i<numRdc; i++) {
        float obs = 5.0 * distribution(generator);
        int restIdx = force->addRdcRestraint(
            2*i, 2*i+1,
            kappa,
            obs,
            tol,
            fc1,
            cut,
            weight
        );
        rest_ids.push_back(restIdx);
    }

    force->addExperiment(rest_ids);
    system.addForce(force);

    vector<Vec3> pos1(positions);

    VerletIntegrator integ(1.0);
    Platform& platform = Platform::getPlatformByName("CUDA");
    Context context(system, integ, platform);

    // compute the eneriy and forces with fc1
    context.setPositions(pos1);
    State state1 = context.getState(State::Energy | State::Forces);
    double energy1 = state1.getPotentialEnergy();
    vector<Vec3> forces1 = state1.getForces();

    // compute the energy and forces with fc2
    for(int index=0; index<numRdc; index++) {
        int i, j, global;
        float kappa=0;
        float obs=0;
        float tol=0;
        float fc=0;
        float cut=0;
        float weight=0;
        force->getRdcRestraintInfo(index, i, j, kappa, obs, tol, fc, cut, weight, global);
        force->updateRdcRestraint(index, i, j, kappa, obs, tol, fc2, cut, weight);
    }
    force->updateParametersInContext(context);

    State state2 = context.getState(State::Energy | State::Forces);
    double energy2 = state2.getPotentialEnergy();
    vector<Vec3> forces2 = state2.getForces();

    // compute the energy and forces with fc0
    for(int index=0; index<numRdc; index++) {
        int i, j, global;
        float kappa=0;
        float obs=0;
        float tol=0;
        float fc=0;
        float cut=0;
        float weight=0;
        force->getRdcRestraintInfo(index, i, j, kappa, obs, tol, fc, cut, weight, global);
        force->updateRdcRestraint(index, i, j, kappa, obs, tol, fc0, cut, weight);
    }
    force->updateParametersInContext(context);

    State state0 = context.getState(State::Energy | State::Forces);
    double energy0 = state0.getPotentialEnergy();
    vector<Vec3> forces0 = state0.getForces();



    // energy with fc2 should be twice with fc1
    ASSERT_EQUAL_TOL(2 * energy1, energy2, 1e-4);

    // energy with fc0 should be zero
    ASSERT_EQUAL_TOL(0.0, energy0, 1e-4);

    // The forces should be twice as high with fc2 than fc1
    for(int i=0; i<numParticles; i++) {
        for(int j=0; j<3; j++) {
            ASSERT_EQUAL_TOL(2*forces1[i][j], forces2[i][j], 1e-3);
        }
    }

    // The forces should be zero with fc0
    for(int i=0; i<numParticles; i++) {
        for(int j=0; j<3; j++) {
            ASSERT_EQUAL_TOL(0.0, forces0[i][j], 1e-3);
        }
    }
}
void testTranslationInvariance(float cut) {
    // Setup RNG
    default_random_engine generator;
    generator.seed(1979);
    normal_distribution<double> distribution(0.0, 1.0);

    System system;

    // Add the particles with random positions
    const int numParticles = 256;
    const int numRdc = 128;
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
    float fc = 10.0;
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
            cut,
            weight
        );
        rest_ids.push_back(restIdx);
    }

    force->addExperiment(rest_ids);
    system.addForce(force);

    vector<Vec3> pos1(positions);
    vector<Vec3> pos2(positions);

    const float pert = 1.0;
    for(int i=0; i<numParticles; i++) {
        pos2[i][0] += pert;
        pos2[i][1] += -pert;
        pos2[i][2] += pert;
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
            ASSERT_EQUAL_TOL(forces1[i][j], forces2[i][j], 1e-2);
        }
    }
}


void testRotationInvariance(float cut) {
    // Setup RNG
    default_random_engine generator;
    generator.seed(1979);
    normal_distribution<double> distribution(0.0, 1.0);

    System system;

    // Add the particles with random positions
    const int numParticles = 256;
    const int numRdc = 128;
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
    float fc = 10.0;
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
            cut,
            weight
        );
        rest_ids.push_back(restIdx);
    }

    force->addExperiment(rest_ids);
    system.addForce(force);

    vector<Vec3> pos1(positions);
    vector<Vec3> pos2(positions);

    // we'll rotate around the z-axis by 90 degrees
    // so x = y_old, y = -x_old
    for(int i=0; i<numParticles; i++) {
        pos2[i][0] = pos1[i][1];
        pos2[i][1] = -pos1[i][0];
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
        ASSERT_EQUAL_TOL(forces1[i][1], forces2[i][0], 1e-3);
        ASSERT_EQUAL_TOL(-forces1[i][0], forces2[i][1], 1e-3);
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
    const int numParticles = 256;
    const int numRdc = 128;
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
    float fc = 10.0;
    float tol = 0.0;
    float kappa = 10000.0;
    float cut = 999999.0;
    vector<int> rest_ids;

    for(int i=0; i<numRdc; i++) {
        float obs = 5.0 * distribution(generator);
        int restIdx = force->addRdcRestraint(
            2*i, 2*i+1,
            kappa,
            obs,
            tol,
            fc,
            cut,
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
            // The tolerance is coarse, as changing the
            // coordinates will change the alignment,
            // which is not included in the force calculation.
            // We ignore any errors that are smaller than
            // 0.5 kJ / mol / nm, as these are common due
            // to the issue just mentioned. Any errors with
            // a larger absolute value should have a relative
            // error below 1%.
            float delta = fd_force - actual_force;
            if(std::abs(delta) > 0.5) {
                ASSERT_EQUAL_TOL(fd_force, actual_force, 1e-2);
            }
        }
    }
}

void testEnergyGoesDown(float cut) {
    // Setup RNG
    default_random_engine generator;
    normal_distribution<double> distribution(0.0, 1.0);
    generator.seed(1979);

    System system;

    // Add the particles with random positions
    const int numParticles = 256;
    const int numRdc = 128;
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
    float fc = 10.0;
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
            cut,
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
    State state1 = context.getState(State::Energy);
    double energy1 = state1.getPotentialEnergy();

    // If we energy minimize for a few steps, the energy should go down
    LocalEnergyMinimizer::minimize(context, 1, 10);
    State state2 = context.getState(State::Energy);
    double energy2 = state2.getPotentialEnergy();
    ASSERT(energy2 < energy1);    
}


int main(int argc, char* argv[]) {
    try {
        registerMeldCudaKernelFactories();
        if (argc > 1) {
            Platform::getPlatformByName("CUDA").setPropertyDefaultValue("CudaPrecision", string(argv[1]));
        } else
        {
            Platform::getPlatformByName("CUDA").setPropertyDefaultValue("CudaPrecision", string("mixed"));
        }
        
    
        testUpdateForceConstants(99999.);
        testUpdateForceConstants(1.0);
        testUpdateForceConstants(0.1);
        testTranslationInvariance(99999.);
        testTranslationInvariance(1.0);
        testTranslationInvariance(0.1);
        testRotationInvariance(99999.);
        testRotationInvariance(1.0);
        testRotationInvariance(0.1);
        testForceMatchesFiniteDifference();
        testEnergyGoesDown(99999.);
        testEnergyGoesDown(1.);
        testEnergyGoesDown(0.1);
    }
    catch(const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}
