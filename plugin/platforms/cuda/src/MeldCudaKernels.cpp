/*
   Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
   All rights reserved
*/


#ifdef WIN32
  #define _USE_MATH_DEFINES // Needed to get M_PI
#endif
#include "MeldCudaKernels.h"
#include "CudaMeldKernelSources.h"
#include "openmm/internal/ContextImpl.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <numeric>
#include <vector>
#include <Eigen/Dense>

#ifdef _MSC_VER
#include <windows.h>
#endif

using namespace MeldPlugin;
using namespace OpenMM;
using namespace std;

#define CHECK_RESULT(result) \
    if (result != CUDA_SUCCESS) { \
        std::stringstream m; \
        m<<errorMessage<<": "<<cu.getErrorString(result)<<" ("<<result<<")"<<" at "<<__FILE__<<":"<<__LINE__; \
        throw OpenMMException(m.str());\
    }


CudaCalcMeldForceKernel::CudaCalcMeldForceKernel(std::string name, const Platform& platform, CudaContext& cu,
                                                 const System& system) :
    CalcMeldForceKernel(name, platform), cu(cu), system(system)
{

    if (cu.getUseDoublePrecision()) {
        cout << "***\n";
        cout << "*** MeldForce does not support double precision.\n";
        cout << "***" << endl;
        throw OpenMMException("MeldForce does not support double precision");
    }

    numDistRestraints = 0;
    numHyperbolicDistRestraints = 0;
    numTorsionRestraints = 0;
    numDistProfileRestraints = 0;
    numRestraints = 0;
    numGroups = 0;
    numCollections = 0;
    largestGroup = 0;
    largestCollection = 0;
    groupsPerBlock = -1;

    distanceRestRParams = NULL;
    distanceRestKParams = NULL;
    distanceRestAtomIndices = NULL;
    distanceRestGlobalIndices = NULL;
    distanceRestForces = NULL;
    hyperbolicDistanceRestRParams = NULL;
    hyperbolicDistanceRestParams = NULL;
    hyperbolicDistanceRestAtomIndices = NULL;
    hyperbolicDistanceRestGlobalIndices = NULL;
    hyperbolicDistanceRestForces = NULL;
    torsionRestParams = NULL;
    torsionRestAtomIndices = NULL;
    torsionRestGlobalIndices = NULL;
    torsionRestForces = NULL;
    distProfileRestAtomIndices = NULL;
    distProfileRestDistRanges = NULL;
    distProfileRestNumBins = NULL;
    distProfileRestParamBounds = NULL;
    distProfileRestParams = NULL;
    distProfileRestScaleFactor = NULL;
    distProfileRestGlobalIndices = NULL;
    distProfileRestForces = NULL;
    torsProfileRestAtomIndices0 = NULL;
    torsProfileRestAtomIndices1 = NULL;
    torsProfileRestNumBins = NULL;
    torsProfileRestParamBounds = NULL;
    torsProfileRestParams0 = NULL;
    torsProfileRestParams1 = NULL;
    torsProfileRestParams2 = NULL;
    torsProfileRestParams3 = NULL;
    torsProfileRestScaleFactor = NULL;
    torsProfileRestGlobalIndices = NULL;
    torsProfileRestForces = NULL;
    restraintEnergies = NULL;
    restraintActive = NULL;
    groupRestraintIndices = NULL;
    groupRestraintIndicesTemp = NULL;
    groupEnergies = NULL;
    groupActive = NULL;
    groupBounds = NULL;
    groupNumActive = NULL;
    collectionGroupIndices = NULL;
    collectionBounds = NULL;
    collectionNumActive = NULL;
    collectionEnergies = NULL;
    collectionEncounteredNaN = NULL;
}

CudaCalcMeldForceKernel::~CudaCalcMeldForceKernel() {
    cu.setAsCurrent();
    delete distanceRestRParams;
    delete distanceRestKParams;
    delete distanceRestAtomIndices;
    delete distanceRestGlobalIndices;
    delete distanceRestForces;
    delete hyperbolicDistanceRestRParams;
    delete hyperbolicDistanceRestParams;
    delete hyperbolicDistanceRestAtomIndices;
    delete hyperbolicDistanceRestGlobalIndices;
    delete hyperbolicDistanceRestForces;
    delete torsionRestParams;
    delete torsionRestAtomIndices;
    delete torsionRestGlobalIndices;
    delete torsionRestForces;
    delete distProfileRestAtomIndices;
    delete distProfileRestDistRanges;
    delete distProfileRestNumBins;
    delete distProfileRestParamBounds;
    delete distProfileRestParams;
    delete distProfileRestScaleFactor;
    delete distProfileRestGlobalIndices;
    delete distProfileRestForces;
    delete torsProfileRestAtomIndices0;
    delete torsProfileRestAtomIndices1;
    delete torsProfileRestNumBins;
    delete torsProfileRestParamBounds;
    delete torsProfileRestParams0;
    delete torsProfileRestParams1;
    delete torsProfileRestParams2;
    delete torsProfileRestParams3;
    delete torsProfileRestScaleFactor;
    delete torsProfileRestGlobalIndices;
    delete torsProfileRestForces;
    delete restraintEnergies;
    delete restraintActive;
    delete groupRestraintIndices;
    delete groupRestraintIndicesTemp;
    delete groupEnergies;
    delete groupActive;
    delete groupBounds;
    delete groupNumActive;
    delete collectionBounds;
    delete collectionNumActive;
    delete collectionEnergies;
    delete collectionEncounteredNaN;
}


void CudaCalcMeldForceKernel::allocateMemory(const MeldForce& force) {
    numDistRestraints = force.getNumDistRestraints();
    numHyperbolicDistRestraints = force.getNumHyperbolicDistRestraints();
    numTorsionRestraints = force.getNumTorsionRestraints();
    numDistProfileRestraints = force.getNumDistProfileRestraints();
    numDistProfileRestParams = force.getNumDistProfileRestParams();
    numTorsProfileRestraints = force.getNumTorsProfileRestraints();
    numTorsProfileRestParams = force.getNumTorsProfileRestParams();
    numRestraints = force.getNumTotalRestraints();
    numGroups = force.getNumGroups();
    numCollections = force.getNumCollections();

    // setup device memory
    if (numDistRestraints > 0) {
        distanceRestRParams        = CudaArray::create<float4> ( cu, numDistRestraints, "distanceRestRParams");
        distanceRestKParams        = CudaArray::create<float>  ( cu, numDistRestraints, "distanceRestKParams");
        distanceRestAtomIndices    = CudaArray::create<int2>    ( cu, numDistRestraints, "distanceRestAtomIndices");
        distanceRestGlobalIndices  = CudaArray::create<int>    ( cu, numDistRestraints, "distanceRestGlobalIndices");
        distanceRestForces         = CudaArray::create<float3> ( cu, numDistRestraints, "distanceRestForces");
    }

    if (numHyperbolicDistRestraints > 0) {
        hyperbolicDistanceRestRParams        = CudaArray::create<float4> ( cu, numHyperbolicDistRestraints, "hyperbolicDistanceRestRParams");
        hyperbolicDistanceRestParams         = CudaArray::create<float4> ( cu, numHyperbolicDistRestraints, "hyperbolicDistanceRestParams");
        hyperbolicDistanceRestAtomIndices    = CudaArray::create<int2>   ( cu, numHyperbolicDistRestraints, "hyperbolicDistanceRestAtomIndices");
        hyperbolicDistanceRestGlobalIndices  = CudaArray::create<int>    ( cu, numHyperbolicDistRestraints, "hyperbolicDistanceRestGlobalIndices");
        hyperbolicDistanceRestForces         = CudaArray::create<float3> ( cu, numHyperbolicDistRestraints, "hyperbolicDistanceRestForces");
    }

    if (numTorsionRestraints > 0) {
        torsionRestParams          = CudaArray::create<float3> (cu, numTorsionRestraints, "torsionRestParams");
        torsionRestAtomIndices     = CudaArray::create<int4>   (cu, numTorsionRestraints, "torsionRestAtomIndices");
        torsionRestGlobalIndices   = CudaArray::create<int>    (cu, numTorsionRestraints, "torsionRestGlobalIndices");
        torsionRestForces          = CudaArray::create<float3> (cu, numTorsionRestraints * 4, "torsionRestForces");
    }

    if (numDistProfileRestraints > 0) {
        distProfileRestAtomIndices = CudaArray::create<int2>   (cu, numDistProfileRestraints, "distProfileRestAtomIndices");
        distProfileRestDistRanges  = CudaArray::create<float2> (cu, numDistProfileRestraints, "distProfileRestDistRanges");
        distProfileRestNumBins     = CudaArray::create<int>    (cu, numDistProfileRestraints, "distProfileRestNumBins");
        distProfileRestParamBounds = CudaArray::create<int2>   (cu, numDistProfileRestraints, "distProfileRestParamBounds");
        distProfileRestParams      = CudaArray::create<float4> (cu, numDistProfileRestParams, "distProfileRestParams");
        distProfileRestScaleFactor = CudaArray::create<float>  (cu, numDistProfileRestraints, "distProfileRestScaleFactor");
        distProfileRestGlobalIndices=CudaArray::create<int>    (cu, numDistProfileRestraints, "distProfileRestGlobalIndices");
        distProfileRestForces      = CudaArray::create<float3> (cu, numDistProfileRestraints, "distProfileRestForces");
    }

    if (numTorsProfileRestraints > 0) {
        torsProfileRestAtomIndices0= CudaArray::create<int4>   (cu, numTorsProfileRestraints, "torsProfileRestAtomIndices0");
        torsProfileRestAtomIndices1= CudaArray::create<int4>   (cu, numTorsProfileRestraints, "torsProfileRestAtomIndices1");
        torsProfileRestNumBins     = CudaArray::create<int>    (cu, numTorsProfileRestraints, "torsProfileRestNumBins");
        torsProfileRestParamBounds = CudaArray::create<int2>   (cu, numTorsProfileRestraints, "torsProfileRestParamBounds");
        torsProfileRestParams0     = CudaArray::create<float4> (cu, numTorsProfileRestParams, "torsProfileRestParams0");
        torsProfileRestParams1     = CudaArray::create<float4> (cu, numTorsProfileRestParams, "torsProfileRestParams1");
        torsProfileRestParams2     = CudaArray::create<float4> (cu, numTorsProfileRestParams, "torsProfileRestParams2");
        torsProfileRestParams3     = CudaArray::create<float4> (cu, numTorsProfileRestParams, "torsProfileRestParams3");
        torsProfileRestScaleFactor = CudaArray::create<float>  (cu, numTorsProfileRestraints, "torsProfileRestScaleFactor");
        torsProfileRestGlobalIndices=CudaArray::create<int>    (cu, numTorsProfileRestraints, "torsProfileRestGlobalIndices");
        torsProfileRestForces      = CudaArray::create<float3> (cu, 8 * numTorsProfileRestraints, "torsProfileRestForces");
    }

    restraintEnergies         = CudaArray::create<float>  ( cu, numRestraints,     "restraintEnergies");
    restraintActive           = CudaArray::create<float>  ( cu, numRestraints,     "restraintActive");
    groupRestraintIndices     = CudaArray::create<int>    ( cu, numRestraints,     "groupRestraintIndices");
    groupRestraintIndicesTemp = CudaArray::create<int>    ( cu, numRestraints,     "groupRestraintIndicesTemp");
    groupEnergies             = CudaArray::create<float>  ( cu, numGroups,         "groupEnergies");
    groupActive               = CudaArray::create<float>  ( cu, numGroups,         "groupActive");
    groupBounds               = CudaArray::create<int2>   ( cu, numGroups,         "groupBounds");
    groupNumActive            = CudaArray::create<int>    ( cu, numGroups,         "groupNumActive");
    collectionGroupIndices    = CudaArray::create<int>    ( cu, numGroups,         "collectionGroupIndices");
    collectionBounds          = CudaArray::create<int2>   ( cu, numCollections,    "collectionBounds");
    collectionNumActive       = CudaArray::create<int>    ( cu, numCollections,    "collectionNumActive");
    collectionEnergies        = CudaArray::create<int>    ( cu, numCollections,    "collectionEnergies");
    collectionEncounteredNaN  = CudaArray::create<int>    ( cu, 1,                 "collectionEncounteredNaN");

    // setup host memory
    h_distanceRestRParams                 = std::vector<float4> (numDistRestraints, make_float4( 0, 0, 0, 0));
    h_distanceRestKParams                 = std::vector<float>  (numDistRestraints, 0);
    h_distanceRestAtomIndices             = std::vector<int2>   (numDistRestraints, make_int2( -1, -1));
    h_distanceRestGlobalIndices           = std::vector<int>    (numDistRestraints, -1);
    h_hyperbolicDistanceRestRParams       = std::vector<float4> (numHyperbolicDistRestraints, make_float4( 0, 0, 0, 0));
    h_hyperbolicDistanceRestParams        = std::vector<float4> (numHyperbolicDistRestraints, make_float4( 0, 0, 0, 0));
    h_hyperbolicDistanceRestAtomIndices   = std::vector<int2>   (numHyperbolicDistRestraints, make_int2( -1, -1));
    h_hyperbolicDistanceRestGlobalIndices = std::vector<int>    (numHyperbolicDistRestraints, -1);
    h_torsionRestParams                   = std::vector<float3> (numTorsionRestraints, make_float3(0, 0, 0));
    h_torsionRestAtomIndices              = std::vector<int4>   (numTorsionRestraints, make_int4(-1,-1,-1,-1));
    h_torsionRestGlobalIndices            = std::vector<int>    (numTorsionRestraints, -1);
    h_distProfileRestAtomIndices          = std::vector<int2>   (numDistProfileRestraints, make_int2(-1, -1));
    h_distProfileRestDistRanges           = std::vector<float2> (numDistProfileRestraints, make_float2(0, 0));
    h_distProfileRestNumBins              = std::vector<int>    (numDistProfileRestraints, -1);
    h_distProileRestParamBounds           = std::vector<int2>   (numDistProfileRestraints, make_int2(-1, -1));
    h_distProfileRestParams               = std::vector<float4> (numDistProfileRestParams, make_float4(0, 0, 0, 0));
    h_distProfileRestScaleFactor          = std::vector<float>  (numDistProfileRestraints, 0);
    h_distProfileRestGlobalIndices        = std::vector<int>    (numDistProfileRestraints, -1);
    h_torsProfileRestAtomIndices0         = std::vector<int4>   (numTorsProfileRestraints, make_int4(-1, -1, -1, -1));
    h_torsProfileRestAtomIndices1         = std::vector<int4>   (numTorsProfileRestraints, make_int4(-1, -1, -1, -1));
    h_torsProfileRestNumBins              = std::vector<int>    (numTorsProfileRestraints, -1);
    h_torsProileRestParamBounds           = std::vector<int2>   (numTorsProfileRestraints, make_int2(-1, -1));
    h_torsProfileRestParams0              = std::vector<float4> (numTorsProfileRestParams, make_float4(0, 0, 0, 0));
    h_torsProfileRestParams1              = std::vector<float4> (numTorsProfileRestParams, make_float4(0, 0, 0, 0));
    h_torsProfileRestParams2              = std::vector<float4> (numTorsProfileRestParams, make_float4(0, 0, 0, 0));
    h_torsProfileRestParams3              = std::vector<float4> (numTorsProfileRestParams, make_float4(0, 0, 0, 0));
    h_torsProfileRestScaleFactor          = std::vector<float>  (numTorsProfileRestraints, 0);
    h_torsProfileRestGlobalIndices        = std::vector<int>    (numTorsProfileRestraints, -1);
    h_groupRestraintIndices               = std::vector<int>    (numRestraints, -1);
    h_groupBounds                         = std::vector<int2>   (numGroups, make_int2( -1, -1));
    h_groupNumActive                      = std::vector<int>    (numGroups, -1);
    h_collectionGroupIndices              = std::vector<int>    (numGroups, -1);
    h_collectionBounds                    = std::vector<int2>   (numCollections, make_int2( -1, -1));
    h_collectionNumActive                 = std::vector<int>    (numCollections, -1);
    h_collectionEncounteredNaN            = std::vector<int>    (1, 0);
}


/**
 * Error checking helper routines
 */

void checkAtomIndex(const int numAtoms, const std::string& restType, const int atomIndex,
                const int restIndex, const int globalIndex) {
    bool bad = false;
    if (atomIndex < 0) {
        bad = true;
    }
    if (atomIndex >= numAtoms) {
        bad = true;
    }
    if (bad) {
        std::stringstream m;
        m<<"Bad index given in "<<restType<<". atomIndex is "<<atomIndex;
        m<<", globalIndex is: "<<globalIndex<<", restraint index is: "<<restIndex;
        throw OpenMMException(m.str());
    }
}


void checkForceConstant(const float forceConst, const std::string& restType,
                        const int restIndex, const int globalIndex) {
    if (forceConst < 0) {
        std::stringstream m;
        m<<"Force constant is < 0 for "<<restType<<" at globalIndex "<<globalIndex<<", restraint index "<<restIndex;
        throw OpenMMException(m.str());
    }
}


void checkDistanceRestraintRs(const float r1, const float r2, const float r3,
                              const float r4, const int restIndex, const int globalIndex) {
    std::stringstream m;
    bool bad = false;
    m<<"Distance restraint has ";

    if (r1 > r2) {
        m<<"r1 > r2. ";
        bad = true;
    } else if (r2 > r3) {
        m<<"r2 > r3. ";
        bad = true;
    } else if (r3 > r4) {
        m<<"r3 > r4. ";
        bad = true;
    }

    if (bad) {
        m<<"Restraint has index "<<restIndex<<" and globalIndex "<<globalIndex<<".";
        throw OpenMMException(m.str());
    }
}


void checkTorsionRestraintAngles(const float phi, const float deltaPhi, const int index, const int globalIndex) {
    std::stringstream m;
    bool bad = false;

    if ((phi < -180.) || (phi > 180.)) {
        m<<"Torsion restraint phi lies outside of [-180, 180]. ";
        bad = true;
    }
    if ((deltaPhi < 0) || (deltaPhi > 180)) {
        m<<"Torsion restraint deltaPhi lies outside of [0, 180]. ";
        bad = true;
    }
    if (bad) {
        m<<"Restraint has index "<<index<<" and globalIndex "<<globalIndex<<".";
        throw OpenMMException(m.str());
    }
}


void checkGroupCollectionIndices(const int num, const std::vector<int>& indices,
                                 std::vector<int>& assigned, const int index,
                                 const std::string& type1, const std::string& type2) {
    std::stringstream m;
    for(std::vector<int>::const_iterator i=indices.begin(); i!=indices.end(); ++i) {
        // make sure we're in range
        if ((*i >= num) || (*i < 0)) {
            m<<type2<<" with index "<<index<<" references "<<type1<<" outside of range[0,"<<(num-1)<<"].";
            throw OpenMMException(m.str());
        }
        // check to see if this restraint is already assigned to another group
        if (assigned[*i] != -1) {
            m<<type1<<" with index "<<(*i)<<" is assinged to more than one "<<type2<<". ";
            m<<type2<<"s are "<<assigned[*i]<<" and ";
            m<<index<<".";
            throw OpenMMException(m.str());
        }
        // otherwise mark this group as belonging to us
        else {
            assigned[*i] = index;
        }
    }
}


void checkNumActive(const std::vector<int>& indices, const int numActive, const int index, const std::string& type) {
    if ( (numActive < 0) || (numActive > indices.size()) ) {
        std::stringstream m;
        m<<type<<" with index "<<index<<" has numActive out of range [0,"<<indices.size()<<"].";
        throw OpenMMException(m.str());
    }
}


void checkAllAssigned(const std::vector<int>& assigned, const std::string& type1, const std::string& type2) {
    for (std::vector<int>::const_iterator i=assigned.begin(); i!=assigned.end(); ++i) {
        if (*i == -1) {
            std::stringstream m;
            int index = std::distance(assigned.begin(), i);
            m<<type1<<" with index "<<index<<" is not assigned to a "<<type2<<".";
            throw OpenMMException(m.str());
        }
    }
}


void CudaCalcMeldForceKernel::setupDistanceRestraints(const MeldForce& force) {
    int numAtoms = system.getNumParticles();
    std::string restType = "distance restraint";
    for (int i=0; i < numDistRestraints; ++i) {
        int atom_i, atom_j, global_index;
        float r1, r2, r3, r4, k;
        force.getDistanceRestraintParams(i, atom_i, atom_j, r1, r2, r3, r4, k, global_index);

        checkAtomIndex(numAtoms, restType, atom_i, i, global_index);
        checkAtomIndex(numAtoms, restType, atom_j, i, global_index);
        checkForceConstant(k, restType, i, global_index);
        checkDistanceRestraintRs(r1, r2, r3, r4, i, global_index);

        h_distanceRestRParams[i] = make_float4(r1, r2, r3, r4);
        h_distanceRestKParams[i] = k;
        h_distanceRestAtomIndices[i] = make_int2(atom_i, atom_j);
        h_distanceRestGlobalIndices[i] = global_index;
    }
}


void CudaCalcMeldForceKernel::setupHyperbolicDistanceRestraints(const MeldForce& force) {
    int numAtoms = system.getNumParticles();
    std::string restType = "hyperbolic distance restraint";
    for (int i=0; i < numHyperbolicDistRestraints; ++i) {
        int atom_i, atom_j, global_index;
        float r1, r2, r3, r4, k1, k2, asymptote;
        force.getHyperbolicDistanceRestraintParams(i, atom_i, atom_j, r1, r2, r3, r4, k1, asymptote, global_index);

        checkAtomIndex(numAtoms, restType, atom_i, i, global_index);
        checkAtomIndex(numAtoms, restType, atom_j, i, global_index);
        checkForceConstant(k1, restType, i, global_index);
        checkForceConstant(asymptote, restType, i, global_index);
        checkDistanceRestraintRs(r1, r2, r3, r4, i, global_index);

        float a = 3 * (r4 - r3) * (r4 - r3);
        float b = -2 * (r4 - r3) * (r4 - r3) * (r4 - r3);
        k2 = 2.0 * asymptote / a;

        h_hyperbolicDistanceRestRParams[i] = make_float4(r1, r2, r3, r4);
        h_hyperbolicDistanceRestParams[i] = make_float4(k1, k2, a, b);
        h_hyperbolicDistanceRestAtomIndices[i] = make_int2(atom_i, atom_j);
        h_hyperbolicDistanceRestGlobalIndices[i] = global_index;
    }
}


void CudaCalcMeldForceKernel::setupTorsionRestraints(const MeldForce& force) {
    int numAtoms = system.getNumParticles();
    std::string restType = "torsion restraint";
    for (int i=0; i < numTorsionRestraints; ++i) {
        int atom_i, atom_j, atom_k, atom_l, globalIndex;
        float phi, deltaPhi, forceConstant;
        force.getTorsionRestraintParams(i, atom_i, atom_j, atom_k, atom_l, phi, deltaPhi, forceConstant, globalIndex);

        checkAtomIndex(numAtoms, restType, atom_i, i, globalIndex);
        checkAtomIndex(numAtoms, restType, atom_j, i, globalIndex);
        checkAtomIndex(numAtoms, restType, atom_k, i, globalIndex);
        checkAtomIndex(numAtoms, restType, atom_l, i, globalIndex);
        checkForceConstant(forceConstant, restType, i, globalIndex);
        checkTorsionRestraintAngles(phi, deltaPhi, i, globalIndex);

        h_torsionRestParams[i] = make_float3(phi, deltaPhi, forceConstant);
        h_torsionRestAtomIndices[i] = make_int4(atom_i, atom_j, atom_k, atom_l);
        h_torsionRestGlobalIndices[i] = globalIndex;
    }
}


void CudaCalcMeldForceKernel::setupDistProfileRestraints(const MeldForce& force) {
    int numAtoms = system.getNumParticles();
    std::string restType = "distance profile restraint";
    int currentParamIndex = 0;
    for (int i=0; i < numDistProfileRestraints; ++i) {
        int thisStart = currentParamIndex;

        int atom1, atom2, nBins, globalIndex;
        float rMin, rMax, scaleFactor;
        std::vector<double> a0, a1, a2, a3;

        force.getDistProfileRestraintParams(i, atom1, atom2, rMin, rMax, nBins,
                a0, a1, a2, a3, scaleFactor, globalIndex);

        checkAtomIndex(numAtoms, restType, atom1, i, globalIndex);
        checkAtomIndex(numAtoms, restType, atom2, i, globalIndex);
        checkForceConstant(scaleFactor, restType, i, globalIndex);

        h_distProfileRestAtomIndices[i] = make_int2(atom1, atom2);
        h_distProfileRestDistRanges[i] = make_float2(rMin, rMax);
        h_distProfileRestNumBins[i] = nBins;
        h_distProfileRestGlobalIndices[i] = globalIndex;
        h_distProfileRestScaleFactor[i] = scaleFactor;

        for (int j=0; j<nBins; ++j) {
            h_distProfileRestParams[currentParamIndex] = make_float4(
                    (float)a0[j],
                    (float)a1[j],
                    (float)a2[j],
                    (float)a3[j]);
            currentParamIndex++;
        }
        int thisEnd = currentParamIndex;
        h_distProileRestParamBounds[i] = make_int2(thisStart, thisEnd);
    }
}

void CudaCalcMeldForceKernel::setupTorsProfileRestraints(const MeldForce& force){
    int numAtoms = system.getNumParticles();
    std::string restType = "torsion profile restraint";
    int currentParamIndex = 0;
    for (int i=0; i < numTorsProfileRestraints; ++i) {
        int thisStart = currentParamIndex;

        int atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, nBins, globalIndex;
        float scaleFactor;
        std::vector<double> a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15;

        force.getTorsProfileRestraintParams(i, atom1, atom2, atom3, atom4,
                atom5, atom6, atom7, atom8, nBins,
                a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15,
                scaleFactor, globalIndex);

        checkAtomIndex(numAtoms, restType, atom1, i, globalIndex);
        checkAtomIndex(numAtoms, restType, atom2, i, globalIndex);
        checkAtomIndex(numAtoms, restType, atom3, i, globalIndex);
        checkAtomIndex(numAtoms, restType, atom4, i, globalIndex);
        checkAtomIndex(numAtoms, restType, atom5, i, globalIndex);
        checkAtomIndex(numAtoms, restType, atom6, i, globalIndex);
        checkAtomIndex(numAtoms, restType, atom7, i, globalIndex);
        checkAtomIndex(numAtoms, restType, atom8, i, globalIndex);
        checkForceConstant(scaleFactor, restType, i, globalIndex);

        h_torsProfileRestAtomIndices0[i] = make_int4(atom1, atom2, atom3, atom4);
        h_torsProfileRestAtomIndices1[i] = make_int4(atom5, atom6, atom7, atom8);
        h_torsProfileRestNumBins[i] = nBins;
        h_torsProfileRestGlobalIndices[i] = globalIndex;
        h_torsProfileRestScaleFactor[i] = scaleFactor;

        for (int j=0; j<nBins*nBins; ++j) {
            h_torsProfileRestParams0[currentParamIndex] = make_float4(
                    (float)a0[j],
                    (float)a1[j],
                    (float)a2[j],
                    (float)a3[j]);
            h_torsProfileRestParams1[currentParamIndex] = make_float4(
                    (float)a4[j],
                    (float)a5[j],
                    (float)a6[j],
                    (float)a7[j]);
            h_torsProfileRestParams2[currentParamIndex] = make_float4(
                    (float)a8[j],
                    (float)a9[j],
                    (float)a10[j],
                    (float)a11[j]);
            h_torsProfileRestParams3[currentParamIndex] = make_float4(
                    (float)a12[j],
                    (float)a13[j],
                    (float)a14[j],
                    (float)a15[j]);
            currentParamIndex++;
        }
        int thisEnd = currentParamIndex;
        h_torsProileRestParamBounds[i] = make_int2(thisStart, thisEnd);
    }
}

void CudaCalcMeldForceKernel::setupGroups(const MeldForce& force) {
    largestGroup = 0;
    std::vector<int> restraintAssigned(numRestraints, -1);
    int start = 0;
    int end = 0;
    for (int i=0; i<numGroups; ++i) {
        std::vector<int> indices;
        int numActive;
        force.getGroupParams(i, indices, numActive);

        checkGroupCollectionIndices(numRestraints, indices, restraintAssigned, i, "Restraint", "Group");
        checkNumActive(indices, numActive, i, "Group");

        int groupSize = indices.size();
        if (groupSize > largestGroup) {
            largestGroup = groupSize;
        }

        end = start + groupSize;
        h_groupNumActive[i] = numActive;
        h_groupBounds[i] = make_int2(start, end);

        for (int j=0; j<indices.size(); ++j) {
            h_groupRestraintIndices[start+j] = indices[j];
        }
        start = end;
    }
    checkAllAssigned(restraintAssigned, "Restraint", "Group");
}


void CudaCalcMeldForceKernel::setupCollections(const MeldForce& force) {
    largestCollection = 0;
    std::vector<int> groupAssigned(numGroups, -1);
    int start=0;
    int end=0;
    for (int i=0; i<numCollections; ++i) {
        std::vector<int> indices;
        int numActive;
        force.getCollectionParams(i, indices, numActive);

        checkGroupCollectionIndices(numGroups, indices, groupAssigned, i, "Group", "Collection");
        checkNumActive(indices, numActive, i, "Collection");

        int collectionSize = indices.size();

        if (collectionSize > largestCollection) {
            largestCollection = collectionSize;
        }

        end = start + collectionSize;
        h_collectionNumActive[i] = numActive;
        h_collectionBounds[i] = make_int2(start, end);
        for (int j=0; j<indices.size(); ++j) {
            h_collectionGroupIndices[start+j] = indices[j];
        }
        start = end;
    }
    checkAllAssigned(groupAssigned, "Group", "Collection");
}


void CudaCalcMeldForceKernel::validateAndUpload() {
    if (numDistRestraints > 0) {
        distanceRestRParams->upload(h_distanceRestRParams);
        distanceRestKParams->upload(h_distanceRestKParams);
        distanceRestAtomIndices->upload(h_distanceRestAtomIndices);
        distanceRestGlobalIndices->upload(h_distanceRestGlobalIndices);
    }

    if (numHyperbolicDistRestraints > 0) {
        hyperbolicDistanceRestRParams->upload(h_hyperbolicDistanceRestRParams);
        hyperbolicDistanceRestParams->upload(h_hyperbolicDistanceRestParams);
        hyperbolicDistanceRestAtomIndices->upload(h_hyperbolicDistanceRestAtomIndices);
        hyperbolicDistanceRestGlobalIndices->upload(h_hyperbolicDistanceRestGlobalIndices);
    }

    if (numTorsionRestraints > 0) {
        torsionRestParams->upload(h_torsionRestParams);
        torsionRestAtomIndices->upload(h_torsionRestAtomIndices);
        torsionRestGlobalIndices->upload(h_torsionRestGlobalIndices);
    }

    if (numDistProfileRestraints > 0) {
        distProfileRestAtomIndices->upload(h_distProfileRestAtomIndices);
        distProfileRestDistRanges->upload(h_distProfileRestDistRanges);
        distProfileRestNumBins->upload(h_distProfileRestNumBins);
        distProfileRestParamBounds->upload(h_distProileRestParamBounds);
        distProfileRestParams->upload(h_distProfileRestParams);
        distProfileRestScaleFactor->upload(h_distProfileRestScaleFactor);
        distProfileRestGlobalIndices->upload(h_distProfileRestGlobalIndices);
    }

    if (numTorsProfileRestraints > 0) {
        torsProfileRestAtomIndices0->upload(h_torsProfileRestAtomIndices0);
        torsProfileRestAtomIndices1->upload(h_torsProfileRestAtomIndices1);
        torsProfileRestNumBins->upload(h_torsProfileRestNumBins);
        torsProfileRestParamBounds->upload(h_torsProileRestParamBounds);
        torsProfileRestParams0->upload(h_torsProfileRestParams0);
        torsProfileRestParams1->upload(h_torsProfileRestParams1);
        torsProfileRestParams2->upload(h_torsProfileRestParams2);
        torsProfileRestParams3->upload(h_torsProfileRestParams3);
        torsProfileRestScaleFactor->upload(h_torsProfileRestScaleFactor);
        torsProfileRestGlobalIndices->upload(h_torsProfileRestGlobalIndices);
    }

    groupRestraintIndices->upload(h_groupRestraintIndices);
    groupBounds->upload(h_groupBounds);
    groupNumActive->upload(h_groupNumActive);
    collectionGroupIndices->upload(h_collectionGroupIndices);
    collectionBounds->upload(h_collectionBounds);
    collectionNumActive->upload(h_collectionNumActive);
}


void CudaCalcMeldForceKernel::initialize(const System& system, const MeldForce& force) {
    cu.setAsCurrent();

    allocateMemory(force);
    setupDistanceRestraints(force);
    setupHyperbolicDistanceRestraints(force);
    setupTorsionRestraints(force);
    setupDistProfileRestraints(force);
    setupTorsProfileRestraints(force);
    setupGroups(force);
    setupCollections(force);
    validateAndUpload();

    std::map<std::string, std::string> replacements;
    std::map<std::string, std::string> defines;
    defines["NUM_ATOMS"] = cu.intToString(cu.getNumAtoms());
    defines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
    replacements["MAXGROUPSIZE"] = cu.intToString(largestGroup);
    replacements["MAXCOLLECTIONSIZE"] = cu.intToString(largestCollection);

    // setup thr maximum number of groups calculated in a single block
    // want to maximize occupancy, but need to ensure that we fit
    // into shared memory
    int sharedSizeGroup = largestGroup * (sizeof(float) + sizeof(int));
    int sharedSizeThreads = 32 * sizeof(float);
    int sharedSize = std::max(sharedSizeGroup, sharedSizeThreads);
    int maxSharedMemory = 48 * 1024;
    groupsPerBlock = std::min(maxSharedMemory / sharedSize, 32);
    if (groupsPerBlock < 1) {
        throw OpenMMException("One of the groups is too large to fit into shared memory.");
    }
    replacements["GROUPSPERBLOCK"] = cu.intToString(groupsPerBlock);

    CUmodule module = cu.createModule(cu.replaceStrings(CudaMeldKernelSources::vectorOps + CudaMeldKernelSources::computeMeld, replacements), defines);
    computeDistRestKernel = cu.getKernel(module, "computeDistRest");
    computeHyperbolicDistRestKernel = cu.getKernel(module, "computeHyperbolicDistRest");
    computeTorsionRestKernel = cu.getKernel(module, "computeTorsionRest");
    computeDistProfileRestKernel = cu.getKernel(module, "computeDistProfileRest");
    computeTorsProfileRestKernel = cu.getKernel(module, "computeTorsProfileRest");
    evaluateAndActivateKernel = cu.getKernel(module, "evaluateAndActivate");
    evaluateAndActivateCollectionsKernel = cu.getKernel(module, "evaluateAndActivateCollections");
    applyGroupsKernel = cu.getKernel(module, "applyGroups");
    applyDistRestKernel = cu.getKernel(module, "applyDistRest");
    applyHyperbolicDistRestKernel = cu.getKernel(module, "applyHyperbolicDistRest");
    applyTorsionRestKernel = cu.getKernel(module, "applyTorsionRest");
    applyDistProfileRestKernel = cu.getKernel(module, "applyDistProfileRest");
    applyTorsProfileRestKernel = cu.getKernel(module, "applyTorsProfileRest");
}


void CudaCalcMeldForceKernel::copyParametersToContext(ContextImpl& context, const MeldForce& force) {
    cu.setAsCurrent();

    setupDistanceRestraints(force);
    setupHyperbolicDistanceRestraints(force);
    setupTorsionRestraints(force);
    setupDistProfileRestraints(force);
    setupTorsProfileRestraints(force);
    setupGroups(force);
    setupCollections(force);
    validateAndUpload();

    // Mark that the current reordering may be invalid.
    cu.invalidateMolecules();
}


double CudaCalcMeldForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    // compute the forces and energies
    if (numDistRestraints > 0) {
        void* distanceArgs[] = {
            &cu.getPosq().getDevicePointer(),
            &distanceRestAtomIndices->getDevicePointer(),
            &distanceRestRParams->getDevicePointer(),
            &distanceRestKParams->getDevicePointer(),
            &distanceRestGlobalIndices->getDevicePointer(),
            &restraintEnergies->getDevicePointer(),
            &distanceRestForces->getDevicePointer(),
            &numDistRestraints};
        cu.executeKernel(computeDistRestKernel, distanceArgs, numDistRestraints);
    }

    if (numHyperbolicDistRestraints > 0) {
        void* hyperbolicDistanceArgs[] = {
            &cu.getPosq().getDevicePointer(),
            &hyperbolicDistanceRestAtomIndices->getDevicePointer(),
            &hyperbolicDistanceRestRParams->getDevicePointer(),
            &hyperbolicDistanceRestParams->getDevicePointer(),
            &hyperbolicDistanceRestGlobalIndices->getDevicePointer(),
            &restraintEnergies->getDevicePointer(),
            &hyperbolicDistanceRestForces->getDevicePointer(),
            &numHyperbolicDistRestraints};
        cu.executeKernel(computeHyperbolicDistRestKernel, hyperbolicDistanceArgs, numHyperbolicDistRestraints);
    }

    if (numTorsionRestraints > 0) {
        void* torsionArgs[] = {
            &cu.getPosq().getDevicePointer(),
            &torsionRestAtomIndices->getDevicePointer(),
            &torsionRestParams->getDevicePointer(),
            &torsionRestGlobalIndices->getDevicePointer(),
            &restraintEnergies->getDevicePointer(),
            &torsionRestForces->getDevicePointer(),
            &numTorsionRestraints};
        cu.executeKernel(computeTorsionRestKernel, torsionArgs, numTorsionRestraints);
    }

    if (numDistProfileRestraints > 0) {
        void * distProfileArgs[] = {
            &cu.getPosq().getDevicePointer(),
            &distProfileRestAtomIndices->getDevicePointer(),
            &distProfileRestDistRanges->getDevicePointer(),
            &distProfileRestNumBins->getDevicePointer(),
            &distProfileRestParams->getDevicePointer(),
            &distProfileRestParamBounds->getDevicePointer(),
            &distProfileRestScaleFactor->getDevicePointer(),
            &distProfileRestGlobalIndices->getDevicePointer(),
            &restraintEnergies->getDevicePointer(),
            &distProfileRestForces->getDevicePointer(),
            &numDistProfileRestraints };
        cu.executeKernel(computeDistProfileRestKernel, distProfileArgs, numDistProfileRestraints);
    }

    if (numTorsProfileRestraints > 0) {
        void * torsProfileArgs[] = {
            &cu.getPosq().getDevicePointer(),
            &torsProfileRestAtomIndices0->getDevicePointer(),
            &torsProfileRestAtomIndices1->getDevicePointer(),
            &torsProfileRestNumBins->getDevicePointer(),
            &torsProfileRestParams0->getDevicePointer(),
            &torsProfileRestParams1->getDevicePointer(),
            &torsProfileRestParams2->getDevicePointer(),
            &torsProfileRestParams3->getDevicePointer(),
            &torsProfileRestParamBounds->getDevicePointer(),
            &torsProfileRestScaleFactor->getDevicePointer(),
            &torsProfileRestGlobalIndices->getDevicePointer(),
            &restraintEnergies->getDevicePointer(),
            &torsProfileRestForces->getDevicePointer(),
            &numTorsProfileRestraints };
        cu.executeKernel(computeTorsProfileRestKernel, torsProfileArgs, numTorsProfileRestraints);
    }

    // now evaluate and active restraints based on groups
    int sharedSizeGroup = largestGroup * (sizeof(float) + sizeof(int));
    int sharedSizeThreads = 32 * sizeof(float);
    int sharedSize = std::max(sharedSizeGroup, sharedSizeThreads);

    void* groupArgs[] = {
        &numGroups,
        &groupNumActive->getDevicePointer(),
        &groupBounds->getDevicePointer(),
        &groupRestraintIndices->getDevicePointer(),
        &groupRestraintIndicesTemp->getDevicePointer(),
        &restraintEnergies->getDevicePointer(),
        &restraintActive->getDevicePointer(),
        &groupEnergies->getDevicePointer()};
    cu.executeKernel(evaluateAndActivateKernel, groupArgs, 32 * numGroups, groupsPerBlock * 32, groupsPerBlock * sharedSize);

    // the kernel will need to be modified if this value is changed
    const int threadsPerCollection = 1024;
    int sharedSizeCollectionEnergies = largestCollection * sizeof(float);
    int sharedSizeCollectionMinMaxBuffer = threadsPerCollection * 2 * sizeof(float);
    int sharedSizeCollectionBinCounts = threadsPerCollection * sizeof(int);
    int sharedSizeCollectionBestBin = sizeof(int);
    int sharedSizeCollection = sharedSizeCollectionEnergies + sharedSizeCollectionMinMaxBuffer +
        sharedSizeCollectionBinCounts + sharedSizeCollectionBestBin;
    // set collectionsEncounteredNaN to zero and upload it
    h_collectionEncounteredNaN[0] = 0;
    collectionEncounteredNaN->upload(h_collectionEncounteredNaN);
    // now evaluate and activate groups based on collections
    void* collArgs[] = {
        &numCollections,
        &collectionNumActive->getDevicePointer(),
        &collectionBounds->getDevicePointer(),
        &collectionGroupIndices->getDevicePointer(),
        &groupEnergies->getDevicePointer(),
        &groupActive->getDevicePointer(),
        &collectionEncounteredNaN->getDevicePointer()};
    cu.executeKernel(evaluateAndActivateCollectionsKernel, collArgs, threadsPerCollection*numCollections, threadsPerCollection, sharedSizeCollection);
    // check if we encountered NaN
    collectionEncounteredNaN->download(h_collectionEncounteredNaN);
    if (h_collectionEncounteredNaN[0]) {
        throw OpenMMException("Encountered NaN when evaluating collections.");
    }

    // Now set the restraints active based on if the groups are active
    void* applyGroupsArgs[] = {
        &groupActive->getDevicePointer(),
        &restraintActive->getDevicePointer(),
        &groupBounds->getDevicePointer(),
        &numGroups};
    cu.executeKernel(applyGroupsKernel, applyGroupsArgs, 32*numGroups, 32);

    // Now apply the forces and energies if the restraints are active
    if (numDistRestraints > 0) {
        void* applyDistRestArgs[] = {
            &cu.getForce().getDevicePointer(),
            &cu.getEnergyBuffer().getDevicePointer(),
            &distanceRestAtomIndices->getDevicePointer(),
            &distanceRestGlobalIndices->getDevicePointer(),
            &distanceRestForces->getDevicePointer(),
            &restraintEnergies->getDevicePointer(),
            &restraintActive->getDevicePointer(),
            &numDistRestraints};
        cu.executeKernel(applyDistRestKernel, applyDistRestArgs, numDistRestraints);
    }

    if (numHyperbolicDistRestraints > 0) {
        void* applyHyperbolicDistRestArgs[] = {
            &cu.getForce().getDevicePointer(),
            &cu.getEnergyBuffer().getDevicePointer(),
            &hyperbolicDistanceRestAtomIndices->getDevicePointer(),
            &hyperbolicDistanceRestGlobalIndices->getDevicePointer(),
            &hyperbolicDistanceRestForces->getDevicePointer(),
            &restraintEnergies->getDevicePointer(),
            &restraintActive->getDevicePointer(),
            &numHyperbolicDistRestraints};
        cu.executeKernel(applyHyperbolicDistRestKernel, applyHyperbolicDistRestArgs, numHyperbolicDistRestraints);
    }

    if (numTorsionRestraints > 0) {
        void* applyTorsionRestArgs[] = {
            &cu.getForce().getDevicePointer(),
            &cu.getEnergyBuffer().getDevicePointer(),
            &torsionRestAtomIndices->getDevicePointer(),
            &torsionRestGlobalIndices->getDevicePointer(),
            &torsionRestForces->getDevicePointer(),
            &restraintEnergies->getDevicePointer(),
            &restraintActive->getDevicePointer(),
            &numTorsionRestraints};
        cu.executeKernel(applyTorsionRestKernel, applyTorsionRestArgs, numTorsionRestraints);
    }

    if (numDistProfileRestraints > 0) {
        void *applyDistProfileRestArgs[] = {
            &cu.getForce().getDevicePointer(),
            &cu.getEnergyBuffer().getDevicePointer(),
            &distProfileRestAtomIndices->getDevicePointer(),
            &distProfileRestGlobalIndices->getDevicePointer(),
            &distProfileRestForces->getDevicePointer(),
            &restraintEnergies->getDevicePointer(),
            &restraintActive->getDevicePointer(),
            &numDistProfileRestraints
        };
        cu.executeKernel(applyDistProfileRestKernel, applyDistProfileRestArgs, numDistProfileRestraints);
    }

    if (numTorsProfileRestraints > 0) {
        void *applyTorsProfileRestArgs[] = {
            &cu.getForce().getDevicePointer(),
            &cu.getEnergyBuffer().getDevicePointer(),
            &torsProfileRestAtomIndices0->getDevicePointer(),
            &torsProfileRestAtomIndices1->getDevicePointer(),
            &torsProfileRestGlobalIndices->getDevicePointer(),
            &torsProfileRestForces->getDevicePointer(),
            &restraintEnergies->getDevicePointer(),
            &restraintActive->getDevicePointer(),
            &numTorsProfileRestraints
        };
        cu.executeKernel(applyTorsProfileRestKernel, applyTorsProfileRestArgs, numTorsProfileRestraints);
    }

    return 0.0;
}


/*
 * RDC Stuff
 */

CudaCalcRdcForceKernel::CudaCalcRdcForceKernel(std::string name, const Platform& platform,
        CudaContext& cu, const System& system) :
    CalcRdcForceKernel(name, platform), cu(cu), system(system) {

    if (cu.getUseDoublePrecision()) {
        cout << "***\n";
        cout << "*** RdcForce does not support double precision.\n";
        cout << "***" << endl;
        throw OpenMMException("RdcForce does not support double precision");
    }

    int numExperiments = 0;
    int numRdcRestraints = 0;

    r = NULL;
    atomExptIndices = NULL;
    lhs = NULL;
    rhs = NULL;
    S = NULL;
    kappa = NULL;
    tolerance = NULL;
    force_const = NULL;
    weight = NULL;
}

CudaCalcRdcForceKernel::~CudaCalcRdcForceKernel() {
    cu.setAsCurrent();
    delete r;
    delete atomExptIndices;
    delete lhs;
    delete rhs;
    delete S;
    delete kappa;
    delete tolerance;
    delete force_const;
    delete weight;
}

void CudaCalcRdcForceKernel::initialize(const System& system, const RdcForce& force) {
    cu.setAsCurrent();

    allocateMemory(force);
    setupRdcRestraints(force);
    validateAndUpload();

    std::map<std::string, std::string> replacements;
    std::map<std::string, std::string> defines;
    defines["NUM_ATOMS"] = cu.intToString(cu.getNumAtoms());
    defines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
    CUmodule module = cu.createModule(cu.replaceStrings(CudaMeldKernelSources::vectorOps + CudaMeldKernelSources::computeRdc, replacements), defines);
    computeRdcPhase1 = cu.getKernel(module, "computeRdcPhase1");
    computeRdcPhase3 = cu.getKernel(module, "computeRdcPhase3");
}

double CudaCalcRdcForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    // TODO: the weights are currently not used

    // Phase 1
    // compute the lhs on the gpu
    void* computePhase1Args[] = {
        &numRdcRestraints,
        &cu.getPosq().getDevicePointer(),
        &atomExptIndices->getDevicePointer(),
        &kappa->getDevicePointer(),
        &r->getDevicePointer(),
        &lhs->getDevicePointer()};
    cu.executeKernel(computeRdcPhase1, computePhase1Args, numRdcRestraints);

    // Phase 2
    // download the lhs, compute S on CPU, upload S back to GPU
    computeRdcPhase2();

    // Phase 3
    // compute the energies and forces on the GPU
    void* computePhase3Args[] = {
        &numRdcRestraints,
        &cu.getPosq().getDevicePointer(),
        &atomExptIndices->getDevicePointer(),
        &kappa->getDevicePointer(),
        &S->getDevicePointer(),
        &rhs->getDevicePointer(),
        &tolerance->getDevicePointer(),
        &force_const->getDevicePointer(),
        &r->getDevicePointer(),
        &lhs->getDevicePointer(),
        &cu.getForce().getDevicePointer(),
        &cu.getEnergyBuffer().getDevicePointer()};
    cu.executeKernel(computeRdcPhase3, computePhase3Args, numRdcRestraints);
    return 0.;
}

void CudaCalcRdcForceKernel::computeRdcPhase2() {
    // Download the lhs from the gpu
    lhs->download(h_lhs);

    // loop over the experiments
    for(int i=0; i<numExperiments; ++i) {
        // get the indices for things in this experiment
        int start = h_experimentBounds[i].x;
        int end = h_experimentBounds[i].y;
        int len = end - start;

        // create wrappers for the correct parts of lhs and rhs
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > lhsWrap(&h_lhs[5 * start], len, 5);
        Eigen::Map<Eigen::VectorXf> rhsWrap(&h_rhs[start], len);
        Eigen::Map<Eigen::VectorXf> SWrap(&h_S[5 * i], 5);


        // solve for S
        SWrap = lhsWrap.jacobiSvd(Eigen::ComputeThinU|Eigen::ComputeThinV).solve(rhsWrap);
    }

    // upload S back up to the gpu
    S->upload(h_S);
}

void CudaCalcRdcForceKernel::copyParametersToContext(ContextImpl& context, const RdcForce& force) {
    cu.setAsCurrent();
    setupRdcRestraints(force);
    validateAndUpload();
    cu.invalidateMolecules();
}

void CudaCalcRdcForceKernel::allocateMemory(const RdcForce& force) {
    numExperiments = force.getNumExperiments();
    numRdcRestraints = force.getNumTotalRestraints();

    /*
     * Allocate device memory
     */
    r = CudaArray::create<float4> (cu, numRdcRestraints, "r");
    atomExptIndices = CudaArray::create<int3> (cu, numRdcRestraints, "atomExptIndices");
    lhs = CudaArray::create<float> (cu, 5 * numRdcRestraints, "lhs");
    rhs = CudaArray::create<float> (cu, numRdcRestraints, "rhs");
    kappa = CudaArray::create<float> (cu, numRdcRestraints, "kappa");
    tolerance = CudaArray::create<float> (cu, numRdcRestraints, "tolerance");
    force_const = CudaArray::create<float> (cu, numRdcRestraints, "force_const");
    weight = CudaArray::create<float> (cu, numRdcRestraints, "weight");
    S = CudaArray::create<float> (cu, 5 * numExperiments, "S");

    /**
     * Allocate host memory
     */
    h_atomExptIndices = std::vector<int3> (numRdcRestraints, make_int3(0, 0, 0));
    h_lhs = std::vector<float> (5 * numRdcRestraints, 0.);
    h_rhs = std::vector<float> (numRdcRestraints, 0.);
    h_kappa = std::vector<float> (numRdcRestraints, 0.);
    h_tolerance = std::vector<float> (numRdcRestraints, 0.);
    h_force_const = std::vector<float> (numRdcRestraints, 0.);
    h_weight = std::vector<float> (numRdcRestraints, 0.);
    h_experimentBounds = std::vector<int2> (numExperiments, make_int2(0, 0));
    h_S = std::vector<float> (5 * numExperiments, 0.);
}

void CudaCalcRdcForceKernel::setupRdcRestraints(const RdcForce& force) {
    int currentIndex = 0;
    // loop over the experiments
    for(int expIndex = 0; expIndex < numExperiments; ++expIndex) {
        int experimentStart = currentIndex;
        std::vector<int> restraintsInExperiment;
        force.getExperimentInfo(expIndex, restraintsInExperiment);

        // loop over the restraints
        for(int withinExpIndex = 0; withinExpIndex < force.getNumRestraints(expIndex); ++withinExpIndex) {
            int currentRestraint = restraintsInExperiment[withinExpIndex];
            int atom1, atom2, globalIndex;
            float kappa, dobs, tolerance, force_const, weight;

            force.getRdcRestraintInfo(currentRestraint, atom1, atom2, kappa, dobs, tolerance,
                    force_const, weight, globalIndex);

            h_atomExptIndices[currentIndex] = make_int3(atom1, atom2, expIndex);
            h_kappa[currentIndex] = kappa;
            h_force_const[currentIndex] = force_const;
            h_weight[currentIndex] = weight;
            h_rhs[currentIndex] = dobs;
            h_tolerance[currentIndex] = tolerance;

            currentIndex++;
        }
        int experimentEnd = currentIndex;
        h_experimentBounds[expIndex] = make_int2(experimentStart, experimentEnd);
    }
}

void CudaCalcRdcForceKernel::validateAndUpload() {
    // todo need to do better validation
    atomExptIndices->upload(h_atomExptIndices);
    lhs->upload(h_lhs);
    rhs->upload(h_rhs);
    tolerance->upload(h_tolerance);
    force_const->upload(h_force_const);
    weight->upload(h_weight);
    kappa->upload(h_kappa);
}
