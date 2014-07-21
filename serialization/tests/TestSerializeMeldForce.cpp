#include "MeldForce.h"
#include "openmm/Platform.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/serialization/XmlSerializer.h"
#include <iostream>
#include <sstream>

using namespace MeldPlugin;
using namespace OpenMM;
using namespace std;

extern "C" void registerMeldSerializationProxies();

void testSerialization() {
    // create MeldForce
    MeldForce force;
    int k = 1.0;
    int restIdx = force.addDistanceRestraint(0, 1, 1.0, 2.0, 3.0, 4.0, k);
    std::vector<int> restIndices(1);
    restIndices[0] = restIdx;
    int groupIdx = force.addGroup(restIndices, 1);
    std::vector<int> groupIndices(1);
    groupIndices[0] = groupIdx;
    force.addCollection(groupIndices, 1);

    // Serialize and then deserialize it.
    stringstream buffer;
    XmlSerializer::serialize<MeldForce>(&force, "Force", buffer);
    MeldForce* copy = XmlSerializer::deserialize<MeldForce>(buffer);

    // Compare the two forces to see if they are identical.
    MeldForce& force2 = *copy;
    ASSERT_EQUAL(force.getNumDistRestraints(), force2.getNumDistRestraints());
    for (int i = 0; i < force.getNumDistRestraints(); i++) {
        int atom1a, atom2a, globalIndexa;
        float r1a, r2a, r3a, r4a, forceConstanta;
        int atom1b, atom2b, globalIndexb;
        float r1b, r2b, r3b, r4b, forceConstantb;
        force.getDistanceRestraintParams(i, atom1a, atom2a, r1a, r2a, r3a, r4a, forceConstanta, globalIndexa);
        force2.getDistanceRestraintParams(i, atom1b, atom2b, r1b, r2b, r3b, r4b, forceConstantb, globalIndexb);
        ASSERT_EQUAL(atom1a, atom1b);
        ASSERT_EQUAL(atom2a, atom2b);
        ASSERT_EQUAL(r1a, r1b);
        ASSERT_EQUAL(r2a, r2b);
        ASSERT_EQUAL(r3a, r3b);
        ASSERT_EQUAL(r4a, r4b);
        ASSERT_EQUAL(forceConstanta, forceConstantb);
        ASSERT_EQUAL(globalIndexa, globalIndexb);
    }

    std::vector<int> indicesA;
    std::vector<int> indicesB;
    int numActiveA, numActiveB;
    force.getGroupParams(0, indicesA, numActiveA);
    force2.getGroupParams(0, indicesB, numActiveB);
    ASSERT_EQUAL(numActiveA, numActiveB);
    ASSERT_EQUAL(indicesA.size(), indicesB.size());
    for (int i = 0; i < indicesA.size(); i++) {
        ASSERT_EQUAL(indicesA[i], indicesB[i]);
    }

    force.getCollectionParams(0, indicesA, numActiveA);
    force2.getCollectionParams(0, indicesB, numActiveB);
    ASSERT_EQUAL(numActiveA, numActiveB);
    ASSERT_EQUAL(indicesA.size(), indicesB.size());
    for (int i = 0; i < indicesA.size(); i++) {
        ASSERT_EQUAL(indicesA[i], indicesB[i]);
    }
}

int main() {
    try {
        registerMeldSerializationProxies();
        testSerialization();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
