/*
   Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
   All rights reserved
*/

#include "RdcForce.h"
#include "openmm/Platform.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/serialization/XmlSerializer.h"
#include <iostream>
#include <sstream>

using namespace MeldPlugin;
using namespace OpenMM;
using namespace std;

extern "C" void registerRdcSerializationProxies();

void testSerialization() {
    // create RdcForce
    RdcForce force;
    int restIdx = force.addRdcRestraint(0, 1, 10.0, 0.0, 10.0, 25000.0, 1.0);
    std::vector<int> rest_ids(1);
    rest_ids[0] = restIdx;
    force.addExperiment(rest_ids);

    // Serialize and then deserialize it.
    stringstream buffer;
    XmlSerializer::serialize<RdcForce>(&force, "Force", buffer);
    RdcForce* copy = XmlSerializer::deserialize<RdcForce>(buffer);

    // Compare the two forces to see if they are identical.
    RdcForce& force2 = *copy;
    ASSERT_EQUAL(force.getNumTotalRestraints(), force2.getNumTotalRestraints());
    for (int i = 0; i < force.getNumTotalRestraints(); i++) {
        int particle1A, particle2A, globalIndexA;
        float kappaA, dObsA, toleranceA, force_constA, weightA;
        int particle1B, particle2B, globalIndexB;
        float kappaB, dObsB, toleranceB, force_constB, weightB;
        force.getRdcRestraintInfo(i, particle1A, particle2A, kappaA, dObsA, toleranceA, force_constA, weightA, globalIndexA);
        force2.getRdcRestraintInfo(i, particle1B, particle2B, kappaB, dObsB, toleranceB, force_constB, weightB, globalIndexB);
        ASSERT_EQUAL(particle1A, particle1B);
        ASSERT_EQUAL(particle2A, particle2B);
        ASSERT_EQUAL(kappaA, kappaB);
        ASSERT_EQUAL(dObsA, dObsB);
        ASSERT_EQUAL(toleranceA, toleranceB);
        ASSERT_EQUAL(force_constA, force_constB);
        ASSERT_EQUAL(weightA, weightB);
        ASSERT_EQUAL(globalIndexA, globalIndexB);
    }

    std::vector<int> indicesA;
    std::vector<int> indicesB;
    force.getExperimentInfo(0, indicesA);
    force2.getExperimentInfo(0, indicesB);
    ASSERT_EQUAL(indicesA.size(), indicesB.size());
    for (int i = 0; i < indicesA.size(); i++) {
        ASSERT_EQUAL(indicesA[i], indicesB[i]);
    }
}

int main() {
    try {
        registerRdcSerializationProxies();
        testSerialization();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
