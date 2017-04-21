/*
   Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
   All rights reserved
*/

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

void testDistRestSerialization() {
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

void testTorsRestSerialization() {
    // create MeldForce
    MeldForce force;
    int k = 1.0;
    int restIdx = force.addTorsionRestraint(0, 1, 2, 3, 0.0, 0.0, k);
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
    ASSERT_EQUAL(force.getNumTorsionRestraints(), force2.getNumTorsionRestraints());
    for (int i = 0; i < force.getNumTorsionRestraints(); i++) {
        int atom1A, atom2A, atom3A, atom4A, globalIndexA;
        float phiA, deltaPhiA, forceConstantA;
        int atom1B, atom2B, atom3B, atom4B, globalIndexB;
        float phiB, deltaPhiB, forceConstantB;
        force.getTorsionRestraintParams(i, atom1A, atom2A, atom3A, atom4A, phiA, deltaPhiA, forceConstantA, globalIndexA);
        force2.getTorsionRestraintParams(i, atom1B, atom2B, atom3B, atom4B, phiB, deltaPhiB, forceConstantB, globalIndexB);
        ASSERT_EQUAL(atom1A, atom1B);
        ASSERT_EQUAL(atom2A, atom2B);
        ASSERT_EQUAL(atom3A, atom3B);
        ASSERT_EQUAL(atom4A, atom4B);
        ASSERT_EQUAL(phiA, phiB);
        ASSERT_EQUAL(deltaPhiA, deltaPhiB);
        ASSERT_EQUAL(forceConstantA, forceConstantB);
        ASSERT_EQUAL(globalIndexA, globalIndexB);
    }
}

void testDistProfRestSerialization() {
    // create MeldForce
    MeldForce force;
    int nBins = 5;
    int restIdx = 0;
    std::vector<double> a(nBins);
    for(int i=0; i<a.size(); i++) {
        a[i] = 1.0;
    }
    restIdx = force.addDistProfileRestraint(0, 1, 1.0, 4.0, nBins, a, a, a, a, 1.0);
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
    ASSERT_EQUAL(force.getNumDistProfileRestraints(), force2.getNumDistProfileRestraints());
    for (int i = 0; i < force.getNumDistProfileRestraints(); i++) {
        int atom1A, atom2A, nBinsA, globalIndexA;
        float rMinA, rMaxA, scaleFactorA;
        std::vector<double> a0A, a1A, a2A, a3A;
        int atom1B, atom2B, nBinsB, globalIndexB;
        float rMinB, rMaxB, scaleFactorB;
        std::vector<double> a0B, a1B, a2B, a3B;
        force.getDistProfileRestraintParams(i, atom1A, atom2A, rMinA, rMaxA, nBinsA, a0A, a1A, a2A, a3A, scaleFactorA, globalIndexA);
        force2.getDistProfileRestraintParams(i, atom1B, atom2B, rMinB, rMaxB, nBinsB, a0B, a1B, a2B, a3B, scaleFactorB, globalIndexB);
        ASSERT_EQUAL(atom1A, atom1B);
        ASSERT_EQUAL(atom2A, atom2B);
        ASSERT_EQUAL(rMinA, rMinB);
        ASSERT_EQUAL(rMaxA, rMaxB);
        for (int j = 0; j < a0A.size(); j++) {
            ASSERT_EQUAL(a0A[j], a0B[j])
        }
        for (int j = 0; j < a1A.size(); j++) {
            ASSERT_EQUAL(a1A[j], a1B[j])
        }
        for (int j = 0; j < a2A.size(); j++) {
            ASSERT_EQUAL(a2A[j], a2B[j])
        }
        for (int j = 0; j < a3A.size(); j++) {
            ASSERT_EQUAL(a3A[j], a3B[j])
        }
        ASSERT_EQUAL(scaleFactorA, scaleFactorB);
        ASSERT_EQUAL(globalIndexA, globalIndexB);
    }
}

void testTorsProfRestSerialization() {
    // create MeldForce
    MeldForce force;
    int nBins = 5;
    int restIdx = 0;
    std::vector<double> a(nBins);
    for(int i=0; i<a.size(); i++) {
        a[i] = 1.0;
    }
    restIdx = force.addTorsProfileRestraint(0, 1, 2, 3, 0, 1, 2, 3, nBins, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, 1.0);
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
    ASSERT_EQUAL(force.getNumTorsProfileRestraints(), force2.getNumTorsProfileRestraints());
    for (int i = 0; i < force.getNumTorsProfileRestraints(); i++) {
        int atom1A, atom2A, atom3A, atom4A, atom5A, atom6A, atom7A, atom8A;
        int nBinsA, globalIndexA;
        float scaleFactorA;
        std::vector<double> a0A, a1A, a2A, a3A, a4A, a5A, a6A, a7A, a8A, a9A, a10A, a11A, a12A, a13A, a14A, a15A;
        int atom1B, atom2B, atom3B, atom4B, atom5B, atom6B, atom7B, atom8B;
        int nBinsB, globalIndexB;
        float scaleFactorB;
        std::vector<double> a0B, a1B, a2B, a3B, a4B, a5B, a6B, a7B, a8B, a9B, a10B, a11B, a12B, a13B, a14B, a15B;
        force.getTorsProfileRestraintParams(i, atom1A, atom2A, atom3A, atom4A, atom5A, atom6A, atom7A, atom8A,
                                            nBinsA, a0A, a1A, a2A, a3A, a4A, a5A, a6A, a7A, a8A, a9A, a10A, a11A, a12A, a13A, a14A, a15A,
                                            scaleFactorA, globalIndexA);
        force2.getTorsProfileRestraintParams(i, atom1B, atom2B, atom3B, atom4B, atom5B, atom6B, atom7B, atom8B,
                                            nBinsB, a0B, a1B, a2B, a3B, a4B, a5B, a6B, a7B, a8B, a9B, a10B, a11B, a12B, a13B, a14B, a15B,
                                            scaleFactorB, globalIndexB);
        ASSERT_EQUAL(atom1A, atom1B);
        ASSERT_EQUAL(atom2A, atom2B);
        ASSERT_EQUAL(atom3A, atom3B);
        ASSERT_EQUAL(atom4A, atom4B);
        ASSERT_EQUAL(atom5A, atom5B);
        ASSERT_EQUAL(atom6A, atom6B);
        ASSERT_EQUAL(atom7A, atom7B);
        ASSERT_EQUAL(atom8A, atom8B);

        for (int j = 0; j < a0A.size(); j++) {
            ASSERT_EQUAL(a0A[j], a0B[j])
        }
        for (int j = 0; j < a1A.size(); j++) {
            ASSERT_EQUAL(a1A[j], a1B[j])
        }
        for (int j = 0; j < a2A.size(); j++) {
            ASSERT_EQUAL(a2A[j], a2B[j])
        }
        for (int j = 0; j < a3A.size(); j++) {
            ASSERT_EQUAL(a3A[j], a3B[j])
        }
        for (int j = 0; j < a4A.size(); j++) {
            ASSERT_EQUAL(a4A[j], a4B[j])
        }
        for (int j = 0; j < a5A.size(); j++) {
            ASSERT_EQUAL(a5A[j], a5B[j])
        }
        for (int j = 0; j < a6A.size(); j++) {
            ASSERT_EQUAL(a6A[j], a6B[j])
        }
        for (int j = 0; j < a7A.size(); j++) {
            ASSERT_EQUAL(a7A[j], a7B[j])
        }
        for (int j = 0; j < a8A.size(); j++) {
            ASSERT_EQUAL(a8A[j], a8B[j])
        }
        for (int j = 0; j < a9A.size(); j++) {
            ASSERT_EQUAL(a9A[j], a9B[j])
        }
        for (int j = 0; j < a10A.size(); j++) {
            ASSERT_EQUAL(a10A[j], a10B[j])
        }
        for (int j = 0; j < a11A.size(); j++) {
            ASSERT_EQUAL(a11A[j], a11B[j])
        }
        for (int j = 0; j < a12A.size(); j++) {
            ASSERT_EQUAL(a12A[j], a12B[j])
        }
        for (int j = 0; j < a13A.size(); j++) {
            ASSERT_EQUAL(a13A[j], a13B[j])
        }
        for (int j = 0; j < a14A.size(); j++) {
            ASSERT_EQUAL(a14A[j], a14B[j])
        }
        for (int j = 0; j < a15A.size(); j++) {
            ASSERT_EQUAL(a15A[j], a15B[j])
        }
        ASSERT_EQUAL(scaleFactorA, scaleFactorB);
        ASSERT_EQUAL(globalIndexA, globalIndexB);
    }
}

void testGMMRestSerialization() {
    // create MeldForce
    MeldForce force;
    int nPairs = 2;
    int nComponents = 1;
    float scale = 1.0;
    std::vector<int> atomIndices = {0, 1, 2, 3};
    std::vector<double> weights = {1.0};
    std::vector<double> means = {1.0, 1.0};
    std::vector<double> precisionOnDiagonals = {1.0, 1.0};
    std::vector<double> precisionOffDiagonals = {0.5};

    // create the restraint, group, and collection
    int restIdx = force.addGMMRestraint(nPairs, nComponents, scale,
                                        atomIndices, weights, means,
                                        precisionOnDiagonals,
                                        precisionOffDiagonals);
    std::vector<int> restIndices = {restIdx};
    int groupIdx = force.addGroup(restIndices, 1);
    std::vector<int> groupIndices = {groupIdx};
    force.addCollection(groupIndices, 1);


    // Serialize and then deserialize it.
    stringstream buffer;
    XmlSerializer::serialize<MeldForce>(&force, "Force", buffer);
    MeldForce* copy = XmlSerializer::deserialize<MeldForce>(buffer);

    // Compare the two forces to see if they are identical.
    MeldForce& force2 = *copy;
    ASSERT_EQUAL(force.getNumGMMRestraints(), force2.getNumGMMRestraints());

    for (int i = 0; i < force.getNumGMMRestraints(); i++) {
        int nPairsA, nPairsB, nComponentsA, nComponentsB;
        int globalIndexA, globalIndexB;
        float scaleA, scaleB;
        std::vector<int> atomIndicesA, atomIndicesB;
        std::vector<double> weightsA, weightsB;
        std::vector<double> meansA, meansB;
        std::vector<double> precisionOnDiagonalsA, precisionOnDiagonalsB;
        std::vector<double> precisionOffDiagonalsA, precisionOffDiagonalsB;

        force.getGMMRestraintParams(i, nPairsA, nComponentsA, scaleA,
                                    atomIndicesA, weightsA, meansA,
                                    precisionOnDiagonalsA,
                                    precisionOffDiagonalsA,
                                    globalIndexA);
        force2.getGMMRestraintParams(i, nPairsB, nComponentsB, scaleB,
                                     atomIndicesB, weightsB, meansB,
                                     precisionOnDiagonalsB,
                                     precisionOffDiagonalsB,
                                     globalIndexB);

        ASSERT_EQUAL(nPairsA, nPairsB);
        ASSERT_EQUAL(nComponentsA, nComponentsB);
        ASSERT_EQUAL(scaleA, scaleB);
        for (int i=0; i<atomIndicesA.size(); ++i) {
            ASSERT_EQUAL(atomIndicesA[i], atomIndicesB[i]);
        }
        for (int i=0; i<weightsA.size(); ++i) {
            ASSERT_EQUAL(weightsA[i], weightsB[i]);
        }
        for (int i=0; i<meansA.size(); ++i) {
            ASSERT_EQUAL(meansA[i], meansB[i]);
        }
        for (int i=0; i<precisionOnDiagonalsA.size(); ++i) {
            ASSERT_EQUAL(precisionOnDiagonalsA[i], precisionOnDiagonalsB[i]);
        }
        for (int i=0; i<precisionOffDiagonalsA.size(); ++i) {
            ASSERT_EQUAL(precisionOffDiagonalsA[i], precisionOffDiagonalsB[i]);
        }
        ASSERT_EQUAL(globalIndexA, globalIndexB);
    }
}

int main() {
    try {
        registerMeldSerializationProxies();
        testDistRestSerialization();
        testTorsRestSerialization();
        testDistProfRestSerialization();
        testTorsProfRestSerialization();
        testGMMRestSerialization();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
