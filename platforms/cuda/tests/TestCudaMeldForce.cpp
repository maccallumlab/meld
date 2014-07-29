#include "MeldForce.h"
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

void testDistRest() {
    // setup system
    const int numParticles = 2;
    System system;
    vector<Vec3> positions(numParticles);
    system.addParticle(1.0);
    system.addParticle(1.0);

    // setup meld force
    MeldForce* force = new MeldForce();
    int k = 1.0;
    int restIdx = force->addDistanceRestraint(0, 1, 1.0, 2.0, 3.0, 4.0, k);
    std::vector<int> restIndices(1);
    restIndices[0] = restIdx;
    int groupIdx = force->addGroup(restIndices, 1);
    std::vector<int> groupIndices(1);
    groupIndices[0] = groupIdx;
    force->addCollection(groupIndices, 1);
    system.addForce(force);

    // setup the context
    VerletIntegrator integ(1.0);
    Platform& platform = Platform::getPlatformByName("CUDA");
    Context context(system, integ, platform);

    // test the flat region
    // set the postitions, compute the forces and energy
    // test to make sure they have the expected values
    positions[0] = Vec3(0.0, 0.0, 0.0);
    positions[1] = Vec3(2.5, 0.0, 0.0);
    context.setPositions(positions);
    State state = context.getState(State::Energy | State::Forces);
    float expectedEnergy = 0.0;
    ASSERT_EQUAL_TOL(expectedEnergy, state.getPotentialEnergy(), 1e-5);

}

void testDistRestChangingParameters() {
    // Create particles
    const int numParticles = 2;
    System system;
    vector<Vec3> positions(numParticles);
    system.addParticle(1.0);
    positions[0] = Vec3(0.0, 0.0, 0.0);
    system.addParticle(1.0);
    positions[1] = Vec3(3.5, 0.0, 0.0);

    // Define distance restraint
    MeldForce* force = new MeldForce();
    float k = 1.0;
    int restIdx = force->addDistanceRestraint(0, 1, 1.0, 2.0, 3.0, 4.0, k);
    std::vector<int> restIndices(1);
    restIndices[0] = restIdx;
    int groupIdx = force->addGroup(restIndices, 1);
    std::vector<int> groupIndices(1);
    groupIndices[0] = groupIdx;
    force->addCollection(groupIndices, 1);
    system.addForce(force);

    // Compute the forces and energy.
    VerletIntegrator integ(1.0);
    Platform& platform = Platform::getPlatformByName("CUDA");
    Context context(system, integ, platform);
    context.setPositions(positions);
    State state = context.getState(State::Energy | State::Forces);

    // See if the energy is correct.
    float expectedEnergy = 0.125;
    ASSERT_EQUAL_TOL(expectedEnergy, state.getPotentialEnergy(), 1e-5);

    // Modify the parameters.
    float k2 = 2.0;
    force->modifyDistanceRestraint(0, 0, 1, 1.0, 2.0, 3.0, 4.0, k2);
    force->updateParametersInContext(context);
    state = context.getState(State::Energy);

    // See if the energy is correct after modifying force const.
    expectedEnergy = 0.25;
    ASSERT_EQUAL_TOL(expectedEnergy, state.getPotentialEnergy(), 1e-5);
}

void testTorsRest() {
    // Create particles
    const int numParticles = 4;
    System system;
    vector<Vec3> positions(numParticles);
    system.addParticle(1.0);
    positions[0] = Vec3(-3.0, -3.0, 0.0);
    system.addParticle(1.0);
    positions[1] = Vec3(-3.0, 0.0, 0.0);
    system.addParticle(1.0);
    positions[2] = Vec3(3.0, 0.0, 0.0);
    system.addParticle(1.0);
    positions[3] = Vec3(3.0, 3.0, 0.0);

    // Define torsion restraint
    MeldForce* force = new MeldForce();
    float k = 1.0;
    int restIdx = force->addTorsionRestraint(0, 1, 2, 3, 0.0, 0.0, k);
    std::vector<int> restIndices(1);
    restIndices[0] = restIdx;
    int groupIdx = force->addGroup(restIndices, 1);
    std::vector<int> groupIndices(1);
    groupIndices[0] = groupIdx;
    force->addCollection(groupIndices, 1);
    system.addForce(force);

    // Compute the forces and energy.
    VerletIntegrator integ(1.0);
    Platform& platform = Platform::getPlatformByName("CUDA");
    Context context(system, integ, platform);
    context.setPositions(positions);
    State state = context.getState(State::Energy | State::Forces);

    // See if the energy is correct.
    float expectedEnergy = 16200;
    ASSERT_EQUAL_TOL(expectedEnergy, state.getPotentialEnergy(), 1e-5);
}

void testDistProfileRest() {
    const int numParticles = 2;
    System system;
    vector<Vec3> positions(numParticles);
    system.addParticle(1.0);
    positions[0] = Vec3(0.0, 0.0, 0.0);
    system.addParticle(1.0);
    positions[1] = Vec3(2.5, 0.0, 0.0);

    MeldForce* force = new MeldForce();
    int nBins = 5;
    int restIdx = 0;
    try {
        std::vector<double> a(nBins);
        for(int i=0; i<a.size(); i++) {
            a[i] = 1.0;
        }
        restIdx = force->addDistProfileRestraint(0, 1, 1.0, 4.0, nBins, a, a, a, a, 1.0);
    }
    catch (std::bad_alloc& ba)
    {
        std::cerr << "bad_alloc caught: " << ba.what() << '\n';
    }
    std::vector<int> restIndices(1);
    restIndices[0] = restIdx;
    int groupIdx = force->addGroup(restIndices, 1);
    std::vector<int> groupIndices(1);
    groupIndices[0] = groupIdx;
    force->addCollection(groupIndices, 1);
    system.addForce(force);

    // Compute the forces and energy.
    VerletIntegrator integ(1.0);
    Platform& platform = Platform::getPlatformByName("CUDA");
    Context context(system, integ, platform);
    context.setPositions(positions);
    State state = context.getState(State::Energy | State::Forces);
    
    // See if the energy is correct.
    float expectedEnergy = 75.8565;
    ASSERT_EQUAL_TOL(expectedEnergy, state.getPotentialEnergy(), 1e-5);
}

void testTorsProfileRest() {
    // Create particles
    const int numParticles = 4;
    System system;
    vector<Vec3> positions(numParticles);
    system.addParticle(1.0);
    positions[0] = Vec3(-3.0, -3.0, 0.0);
    system.addParticle(1.0);
    positions[1] = Vec3(-3.0, 0.0, 0.0);
    system.addParticle(1.0);
    positions[2] = Vec3(3.0, 0.0, 0.0);
    system.addParticle(1.0);
    positions[3] = Vec3(3.0, 3.0, 0.0);

    // Define torsion restraint
    MeldForce* force = new MeldForce();
    int nBins = 5;
    int restIdx = 0;
    try {
        std::vector<double> a(nBins);
        for(int i=0; i<a.size(); i++) {
            a[i] = 1.0;
        }
        restIdx = force->addTorsProfileRestraint(0, 1, 2, 3, 0, 1, 2, 3, nBins, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, 1.0);
    }
    catch (std::bad_alloc& ba)
    {
        std::cerr << "bad_alloc caught: " << ba.what() << '\n';
    }
    std::vector<int> restIndices(1);
    restIndices[0] = restIdx;
    int groupIdx = force->addGroup(restIndices, 1);
    std::vector<int> groupIndices(1);
    groupIndices[0] = groupIdx;
    force->addCollection(groupIndices, 1);
    system.addForce(force);

    // Compute the forces and energy.
    VerletIntegrator integ(1.0);
    Platform& platform = Platform::getPlatformByName("CUDA");
    Context context(system, integ, platform);
    context.setPositions(positions);
    State state = context.getState(State::Energy | State::Forces);

    // See if the energy is correct.
    float expectedEnergy = 1.0;
    ASSERT_EQUAL_TOL(expectedEnergy, state.getPotentialEnergy(), 1e-5);
}

int main(int argc, char* argv[]) {
    try {
        registerMeldCudaKernelFactories();
        if (argc > 1)
            Platform::getPlatformByName("CUDA").setPropertyDefaultValue("CudaPrecision", string(argv[1]));
        testDistRest();
        testDistRestChangingParameters();
        testTorsRest();
        testDistProfileRest();
        testTorsProfileRest();
    }
    catch(const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}
