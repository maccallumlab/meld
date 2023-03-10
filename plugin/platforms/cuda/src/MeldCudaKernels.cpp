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
#include "openmm/cuda/CudaForceInfo.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <numeric>
#include <vector>
#include <iostream>
#include <Eigen/Dense>

#ifdef _MSC_VER
#include <windows.h>
#endif

using namespace MeldPlugin;
using namespace OpenMM;
using namespace std;

#define CHECK_RESULT(result)                                                            \
    if (result != CUDA_SUCCESS)                                                         \
    {                                                                                   \
        std::stringstream m;                                                            \
        m << errorMessage << ": " << cu.getErrorString(result) << " (" << result << ")" \
          << " at " << __FILE__ << ":" << __LINE__;                                     \
        throw OpenMMException(m.str());                                                 \
    }

class CudaMeldForceInfo : public CudaForceInfo
{
public:
    std::vector<std::pair<int, int>> bonds;

    CudaMeldForceInfo(const MeldForce &force) : force(force)
    {
        bonds = force.getBondedParticles();
    }

    int getNumParticleGroups() override
    {
        return bonds.size();
    }

    void getParticlesInGroup(int index, vector<int> &particles) override
    {
        particles.clear();
        particles.push_back(bonds[index].first);
        particles.push_back(bonds[index].second);
    }

    bool areParticlesIdentical(int particle1, int particle2) override
    {
        if (force.containsParticle(particle1) || force.containsParticle(particle2))
            return false;
        return true;
    }

    bool areGroupsIdentical(int group1, int group2) override
    {
        return false;
    }

private:
    const MeldForce &force;
};

CudaCalcMeldForceKernel::CudaCalcMeldForceKernel(std::string name, const Platform &platform, CudaContext &cu,
                                                 const System &system) : CalcMeldForceKernel(name, platform), cu(cu), system(system)
{
    if (cu.getUseDoublePrecision())
    {
        cout << "***\n";
        cout << "*** MeldForce does not support double precision.\n";
        cout << "***" << endl;
        throw OpenMMException("MeldForce does not support double precision");
    }

    numRDCRestraints = 0;
    numRDCAlignments = 0;
    rdcScaleFactor = 0.0;
    numDistRestraints = 0;
    numHyperbolicDistRestraints = 0;
    numTorsionRestraints = 0;
    numDistProfileRestraints = 0;
    numGMMRestraints = 0;
    numGridPotentials = 0;
    numGridPotentialRestraints = 0;
    numGridPotentialGrids = make_int3(0,0,0);
    numGridPotentialAtoms = 0;
    numRestraints = 0;
    numGroups = 0;
    numCollections = 0;
    largestGroup = 0;
    largestCollection = 0;
    groupsPerBlock = -1;

    rdcRestAlignments = nullptr;
    rdcRestParams1 = nullptr;
    rdcRestParams2 = nullptr;
    rdcRestAtomIndices = nullptr;
    rdcRestGlobalIndices = nullptr;
    rdcRestForces = nullptr;
    rdcRestAlignmentComponents = nullptr;
    rdcRestDerivs = nullptr;
    rdcRestDerivIndices = nullptr;
    distanceRestRParams = nullptr;
    distanceRestKParams = nullptr;
    distanceRestAtomIndices = nullptr;
    distanceRestGlobalIndices = nullptr;
    distanceRestForces = nullptr;
    hyperbolicDistanceRestRParams = nullptr;
    hyperbolicDistanceRestParams = nullptr;
    hyperbolicDistanceRestAtomIndices = nullptr;
    hyperbolicDistanceRestGlobalIndices = nullptr;
    hyperbolicDistanceRestForces = nullptr;
    torsionRestParams = nullptr;
    torsionRestAtomIndices = nullptr;
    torsionRestGlobalIndices = nullptr;
    torsionRestForces = nullptr;
    distProfileRestAtomIndices = nullptr;
    distProfileRestDistRanges = nullptr;
    distProfileRestNumBins = nullptr;
    distProfileRestParamBounds = nullptr;
    distProfileRestParams = nullptr;
    distProfileRestScaleFactor = nullptr;
    distProfileRestGlobalIndices = nullptr;
    distProfileRestForces = nullptr;
    torsProfileRestAtomIndices0 = nullptr;
    torsProfileRestAtomIndices1 = nullptr;
    torsProfileRestNumBins = nullptr;
    torsProfileRestParamBounds = nullptr;
    torsProfileRestParams0 = nullptr;
    torsProfileRestParams1 = nullptr;
    torsProfileRestParams2 = nullptr;
    torsProfileRestParams3 = nullptr;
    torsProfileRestScaleFactor = nullptr;
    torsProfileRestGlobalIndices = nullptr;
    torsProfileRestForces = nullptr;
    gmmParams = nullptr;
    gmmOffsets = nullptr;
    gmmAtomIndices = nullptr;
    gmmData = nullptr;
    gmmForces = nullptr;
    gridPotentials = nullptr;
    gridPotentialgridx = nullptr;
    gridPotentialgridy = nullptr;
    gridPotentialgridz = nullptr;
    gridPotentialRestGridPosx = nullptr;
    gridPotentialRestGridPosy = nullptr;
    gridPotentialRestGridPosz = nullptr;
    gridPotentialRestAtomIndices = nullptr;
    gridPotentialRestAtomList = nullptr;
    gridPotentialRestWeights = nullptr;
    gridPotentialRestForces = nullptr;
    gridPotentialRestGlobalIndices = nullptr;
    restraintEnergies = nullptr;
    restraintActive = nullptr;
    groupRestraintIndices = nullptr;
    groupEnergies = nullptr;
    groupActive = nullptr;
    groupBounds = nullptr;
    groupNumActive = nullptr;
    collectionGroupIndices = nullptr;
    collectionBounds = nullptr;
    collectionNumActive = nullptr;
    collectionEnergies = nullptr;
}

CudaCalcMeldForceKernel::~CudaCalcMeldForceKernel()
{
    cu.setAsCurrent();
    delete rdcRestAlignments;
    delete rdcRestParams1;
    delete rdcRestParams2;
    delete rdcRestAtomIndices;
    delete rdcRestGlobalIndices;
    delete rdcRestForces;
    delete rdcRestAlignmentComponents;
    delete rdcRestDerivs;
    delete rdcRestDerivIndices;
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
    delete gmmParams;
    delete gmmOffsets;
    delete gmmAtomIndices;
    delete gmmData;
    delete gmmForces;
    delete gridPotentials;
    delete gridPotentialgridx;
    delete gridPotentialgridy;
    delete gridPotentialgridz;
    delete gridPotentialRestGridPosx;
    delete gridPotentialRestGridPosy;
    delete gridPotentialRestGridPosz;
    delete gridPotentialRestAtomIndices;
    delete gridPotentialRestAtomList;
    delete gridPotentialRestWeights;
    delete gridPotentialRestForces;
    delete gridPotentialRestGlobalIndices;
    delete restraintEnergies;
    delete restraintActive;
    delete groupRestraintIndices;
    delete groupEnergies;
    delete groupActive;
    delete groupBounds;
    delete groupNumActive;
    delete collectionBounds;
    delete collectionNumActive;
    delete collectionEnergies;
}

void CudaCalcMeldForceKernel::allocateMemory(const MeldForce &force)
{
    numRDCAlignments = force.getNumRDCAlignments();
    numRDCRestraints = force.getNumRDCRestraints();
    numDistRestraints = force.getNumDistRestraints();
    numHyperbolicDistRestraints = force.getNumHyperbolicDistRestraints();
    numTorsionRestraints = force.getNumTorsionRestraints();
    numDistProfileRestraints = force.getNumDistProfileRestraints();
    numDistProfileRestParams = force.getNumDistProfileRestParams();
    numTorsProfileRestraints = force.getNumTorsProfileRestraints();
    numTorsProfileRestParams = force.getNumTorsProfileRestParams();
    numGMMRestraints = force.getNumGMMRestraints();
    numGridPotentials = force.getNumGridPotentials();
    numGridPotentialRestraints = force.getNumGridPotentialRestraints();
    numGridPotentialGrids = calcNumGrids(force);
    numGridPotentialAtoms = calcNumGridPotentialAtoms(force);
    numRestraints = force.getNumTotalRestraints();
    numGroups = force.getNumGroups();
    numCollections = force.getNumCollections();
    
    // setup device memory
    if (numRDCRestraints > 0)
    {
        rdcRestParams1 = CudaArray::create<float2>(cu, numRDCRestraints, "rdcRestParams1");
        rdcRestParams2 = CudaArray::create<float3>(cu, numRDCRestraints, "rdcRestParams2");
        rdcRestAlignments = CudaArray::create<int>(cu, numRDCRestraints, "rdcRestAlignments");
        rdcRestAtomIndices = CudaArray::create<int2>(cu, numRDCRestraints, "rdcRestAtomIndices");
        rdcRestGlobalIndices = CudaArray::create<int>(cu, numRDCRestraints, "rdcRestGlobalIndices");
        rdcRestForces = CudaArray::create<float3>(cu, numRDCRestraints, "rdcRestForces");
        rdcRestAlignmentComponents = CudaArray::create<float>(cu, 5 * numRDCAlignments, "rdcRestAlignmentComponents");
        rdcRestDerivs = CudaArray::create<float>(cu, 5 * numRDCRestraints, "rdcRestDerivs");
        rdcRestDerivIndices = CudaArray::create<int>(cu, 5 * numRDCAlignments, "rdcRestDerivIndices");
    }
    if (numRDCAlignments > 0)
    {
        rdcRestDerivIndices = CudaArray::create<int>(cu, 5 * numRDCAlignments, "rdcRestDerivIndices");
    }

    if (numDistRestraints > 0)
    {
        distanceRestRParams = CudaArray::create<float4>(cu, numDistRestraints, "distanceRestRParams");
        distanceRestKParams = CudaArray::create<float>(cu, numDistRestraints, "distanceRestKParams");
        distanceRestAtomIndices = CudaArray::create<int2>(cu, numDistRestraints, "distanceRestAtomIndices");
        distanceRestGlobalIndices = CudaArray::create<int>(cu, numDistRestraints, "distanceRestGlobalIndices");
        distanceRestForces = CudaArray::create<float3>(cu, numDistRestraints, "distanceRestForces");
    }

    if (numHyperbolicDistRestraints > 0)
    {
        hyperbolicDistanceRestRParams = CudaArray::create<float4>(cu, numHyperbolicDistRestraints, "hyperbolicDistanceRestRParams");
        hyperbolicDistanceRestParams = CudaArray::create<float4>(cu, numHyperbolicDistRestraints, "hyperbolicDistanceRestParams");
        hyperbolicDistanceRestAtomIndices = CudaArray::create<int2>(cu, numHyperbolicDistRestraints, "hyperbolicDistanceRestAtomIndices");
        hyperbolicDistanceRestGlobalIndices = CudaArray::create<int>(cu, numHyperbolicDistRestraints, "hyperbolicDistanceRestGlobalIndices");
        hyperbolicDistanceRestForces = CudaArray::create<float3>(cu, numHyperbolicDistRestraints, "hyperbolicDistanceRestForces");
    }

    if (numTorsionRestraints > 0)
    {
        torsionRestParams = CudaArray::create<float3>(cu, numTorsionRestraints, "torsionRestParams");
        torsionRestAtomIndices = CudaArray::create<int4>(cu, numTorsionRestraints, "torsionRestAtomIndices");
        torsionRestGlobalIndices = CudaArray::create<int>(cu, numTorsionRestraints, "torsionRestGlobalIndices");
        torsionRestForces = CudaArray::create<float3>(cu, numTorsionRestraints * 4, "torsionRestForces");
    }

    if (numDistProfileRestraints > 0)
    {
        distProfileRestAtomIndices = CudaArray::create<int2>(cu, numDistProfileRestraints, "distProfileRestAtomIndices");
        distProfileRestDistRanges = CudaArray::create<float2>(cu, numDistProfileRestraints, "distProfileRestDistRanges");
        distProfileRestNumBins = CudaArray::create<int>(cu, numDistProfileRestraints, "distProfileRestNumBins");
        distProfileRestParamBounds = CudaArray::create<int2>(cu, numDistProfileRestraints, "distProfileRestParamBounds");
        distProfileRestParams = CudaArray::create<float4>(cu, numDistProfileRestParams, "distProfileRestParams");
        distProfileRestScaleFactor = CudaArray::create<float>(cu, numDistProfileRestraints, "distProfileRestScaleFactor");
        distProfileRestGlobalIndices = CudaArray::create<int>(cu, numDistProfileRestraints, "distProfileRestGlobalIndices");
        distProfileRestForces = CudaArray::create<float3>(cu, numDistProfileRestraints, "distProfileRestForces");
    }

    if (numTorsProfileRestraints > 0)
    {
        torsProfileRestAtomIndices0 = CudaArray::create<int4>(cu, numTorsProfileRestraints, "torsProfileRestAtomIndices0");
        torsProfileRestAtomIndices1 = CudaArray::create<int4>(cu, numTorsProfileRestraints, "torsProfileRestAtomIndices1");
        torsProfileRestNumBins = CudaArray::create<int>(cu, numTorsProfileRestraints, "torsProfileRestNumBins");
        torsProfileRestParamBounds = CudaArray::create<int2>(cu, numTorsProfileRestraints, "torsProfileRestParamBounds");
        torsProfileRestParams0 = CudaArray::create<float4>(cu, numTorsProfileRestParams, "torsProfileRestParams0");
        torsProfileRestParams1 = CudaArray::create<float4>(cu, numTorsProfileRestParams, "torsProfileRestParams1");
        torsProfileRestParams2 = CudaArray::create<float4>(cu, numTorsProfileRestParams, "torsProfileRestParams2");
        torsProfileRestParams3 = CudaArray::create<float4>(cu, numTorsProfileRestParams, "torsProfileRestParams3");
        torsProfileRestScaleFactor = CudaArray::create<float>(cu, numTorsProfileRestraints, "torsProfileRestScaleFactor");
        torsProfileRestGlobalIndices = CudaArray::create<int>(cu, numTorsProfileRestraints, "torsProfileRestGlobalIndices");
        torsProfileRestForces = CudaArray::create<float3>(cu, 8 * numTorsProfileRestraints, "torsProfileRestForces");
    }

    if (numGMMRestraints > 0)
    {
        gmmParams = CudaArray::create<int4>(cu, numGMMRestraints, "gmmParams");
        gmmOffsets = CudaArray::create<int2>(cu, numGMMRestraints, "gmmOffsets");
        gmmForces = CudaArray::create<float3>(cu, calcSizeGMMAtomIndices(force), "gmmForces");
        gmmAtomIndices = CudaArray::create<float>(cu, calcSizeGMMAtomIndices(force), "gmmAtomIndices");
        gmmData = CudaArray::create<float>(cu, calcSizeGMMData(force), "gmmData");
    }

    if (numGridPotentials > 0)
    {
        gridPotentials = CudaArray::create<float>(cu, numGridPotentials * calcNumGrids(force).x * calcNumGrids(force).y * calcNumGrids(force).z, "gridPotentials");
        gridPotentialgridx = CudaArray::create<float>(cu, calcNumGrids(force).x, "gridPotentialgridx");
        gridPotentialgridy = CudaArray::create<float>(cu, calcNumGrids(force).y, "gridPotentialgridy");
        gridPotentialgridz = CudaArray::create<float>(cu, calcNumGrids(force).z, "gridPotentialgridz");        
    }

    if (numGridPotentialRestraints > 0) {
        gridPotentialRestAtomIndices            = CudaArray::create<int>    (cu,  calcNumGridPotentialAtoms(force),        "gridPotentialRestAtomIndices");
        gridPotentialRestGridPosx               = CudaArray::create<float>  (cu,  calcNumGrids(force).x,          "gridPotentialRestGridPos");
        gridPotentialRestGridPosy               = CudaArray::create<float>  (cu,  calcNumGrids(force).y,          "gridPotentialRestGridPos");
        gridPotentialRestGridPosz               = CudaArray::create<float>  (cu,  calcNumGrids(force).z,          "gridPotentialRestGridPos");
        gridPotentialRestMu                     = CudaArray::create<float>  (cu,  numGridPotentialRestraints * calcNumGrids(force).x * calcNumGrids(force).y * calcNumGrids(force).z,   "gridPotentialRestMu");
        gridPotentialRestWeights                = CudaArray::create<float>  (cu,  calcNumGridPotentialAtoms(force),        "gridPotentialRestWeights");
        gridPotentialRestAtomList               = CudaArray::create<int>    (cu,  numGridPotentialRestraints+1,            "gridPotentialRestAtomList");
        gridPotentialRestGlobalIndices          = CudaArray::create<int>    (cu,  numGridPotentialRestraints,              "gridPotentialRestGlobalIndices");
        gridPotentialRestForces                 = CudaArray::create<float3> (cu,  calcNumGridPotentialAtoms(force),        "gridPotentialRestForces");
    }

    restraintEnergies = CudaArray::create<float>(cu, numRestraints, "restraintEnergies");
    restraintActive = CudaArray::create<float>(cu, numRestraints, "restraintActive");
    groupRestraintIndices = CudaArray::create<int>(cu, numRestraints, "groupRestraintIndices");
    groupEnergies = CudaArray::create<float>(cu, numGroups, "groupEnergies");
    groupActive = CudaArray::create<float>(cu, numGroups, "groupActive");
    groupBounds = CudaArray::create<int2>(cu, numGroups, "groupBounds");
    groupNumActive = CudaArray::create<int>(cu, numGroups, "groupNumActive");
    collectionGroupIndices = CudaArray::create<int>(cu, numGroups, "collectionGroupIndices");
    collectionBounds = CudaArray::create<int2>(cu, numCollections, "collectionBounds");
    collectionNumActive = CudaArray::create<int>(cu, numCollections, "collectionNumActive");
    collectionEnergies = CudaArray::create<int>(cu, numCollections, "collectionEnergies");

    // setup host memory
    h_rdcRestParams1 = std::vector<float2>(numRDCRestraints, make_float2(0, 0));
    h_rdcRestParams2 = std::vector<float3>(numRDCRestraints, make_float3(0, 0, 0));
    h_rdcRestAlignments = std::vector<int>(numRDCRestraints, -1);
    h_rdcRestAtomIndices = std::vector<int2>(numRDCRestraints, make_int2(-1, -1));
    h_rdcRestGlobalIndices = std::vector<int>(numRDCRestraints, -1);
    h_rdcRestAlignmentComponents = std::vector<float>(5 * numRDCAlignments, 0.0);
    h_rdcRestDerivIndices = std::vector<int>(5 * numRDCAlignments, 0);
    h_distanceRestRParams = std::vector<float4>(numDistRestraints, make_float4(0, 0, 0, 0));
    h_distanceRestKParams = std::vector<float>(numDistRestraints, 0);
    h_distanceRestAtomIndices = std::vector<int2>(numDistRestraints, make_int2(-1, -1));
    h_distanceRestGlobalIndices = std::vector<int>(numDistRestraints, -1);
    h_hyperbolicDistanceRestRParams = std::vector<float4>(numHyperbolicDistRestraints, make_float4(0, 0, 0, 0));
    h_hyperbolicDistanceRestParams = std::vector<float4>(numHyperbolicDistRestraints, make_float4(0, 0, 0, 0));
    h_hyperbolicDistanceRestAtomIndices = std::vector<int2>(numHyperbolicDistRestraints, make_int2(-1, -1));
    h_hyperbolicDistanceRestGlobalIndices = std::vector<int>(numHyperbolicDistRestraints, -1);
    h_torsionRestParams = std::vector<float3>(numTorsionRestraints, make_float3(0, 0, 0));
    h_torsionRestAtomIndices = std::vector<int4>(numTorsionRestraints, make_int4(-1, -1, -1, -1));
    h_torsionRestGlobalIndices = std::vector<int>(numTorsionRestraints, -1);
    h_distProfileRestAtomIndices = std::vector<int2>(numDistProfileRestraints, make_int2(-1, -1));
    h_distProfileRestDistRanges = std::vector<float2>(numDistProfileRestraints, make_float2(0, 0));
    h_distProfileRestNumBins = std::vector<int>(numDistProfileRestraints, -1);
    h_distProileRestParamBounds = std::vector<int2>(numDistProfileRestraints, make_int2(-1, -1));
    h_distProfileRestParams = std::vector<float4>(numDistProfileRestParams, make_float4(0, 0, 0, 0));
    h_distProfileRestScaleFactor = std::vector<float>(numDistProfileRestraints, 0);
    h_distProfileRestGlobalIndices = std::vector<int>(numDistProfileRestraints, -1);
    h_torsProfileRestAtomIndices0 = std::vector<int4>(numTorsProfileRestraints, make_int4(-1, -1, -1, -1));
    h_torsProfileRestAtomIndices1 = std::vector<int4>(numTorsProfileRestraints, make_int4(-1, -1, -1, -1));
    h_torsProfileRestNumBins = std::vector<int>(numTorsProfileRestraints, -1);
    h_torsProileRestParamBounds = std::vector<int2>(numTorsProfileRestraints, make_int2(-1, -1));
    h_torsProfileRestParams0 = std::vector<float4>(numTorsProfileRestParams, make_float4(0, 0, 0, 0));
    h_torsProfileRestParams1 = std::vector<float4>(numTorsProfileRestParams, make_float4(0, 0, 0, 0));
    h_torsProfileRestParams2 = std::vector<float4>(numTorsProfileRestParams, make_float4(0, 0, 0, 0));
    h_torsProfileRestParams3 = std::vector<float4>(numTorsProfileRestParams, make_float4(0, 0, 0, 0));
    h_torsProfileRestScaleFactor = std::vector<float>(numTorsProfileRestraints, 0);
    h_torsProfileRestGlobalIndices = std::vector<int>(numTorsProfileRestraints, -1);
    h_gmmParams = std::vector<int4>(numGMMRestraints, make_int4(0, 0, 0, 0));
    h_gmmOffsets = std::vector<int2>(numGMMRestraints, make_int2(0, 0));
    h_gmmAtomIndices = std::vector<int>(calcSizeGMMAtomIndices(force), 0);
    h_gmmData = std::vector<float>(calcSizeGMMData(force), 0);
    h_gridPotentials = std::vector<float>(numGridPotentials * calcNumGrids(force).x * calcNumGrids(force).y * calcNumGrids(force).z,0);
    h_gridPotentialgridx = std::vector<float>(calcNumGrids(force).x, 0);
    h_gridPotentialgridy = std::vector<float>(calcNumGrids(force).y, 0);
    h_gridPotentialgridz = std::vector<float>(calcNumGrids(force).z, 0);   
    h_gridPotentialRestGridPosx                        = std::vector<float>  (calcNumGrids(force).x, 0);
    h_gridPotentialRestGridPosy                        = std::vector<float>  (calcNumGrids(force).y, 0);
    h_gridPotentialRestGridPosz                        = std::vector<float>  (calcNumGrids(force).z, 0);
    h_gridPotentialRestMu                              = std::vector<float>  (numGridPotentialRestraints * calcNumGrids(force).x * calcNumGrids(force).y * calcNumGrids(force).z, 0);
    h_gridPotentialRestAtomIndices                     = std::vector<int>    (calcNumGridPotentialAtoms(force), -1);
    h_gridPotentialRestAtomList                        = std::vector<int>    (numGridPotentialRestraints+1, -1);
    h_gridPotentialRestWeights                        = std::vector<float>  (calcNumGridPotentialAtoms(force), 0);
    h_gridPotentialRestGlobalIndices                   = std::vector<int>    (numGridPotentialRestraints, -1);
    h_groupRestraintIndices = std::vector<int>(numRestraints, -1);
    h_groupBounds = std::vector<int2>(numGroups, make_int2(-1, -1));
    h_groupNumActive = std::vector<int>(numGroups, -1);
    h_collectionGroupIndices = std::vector<int>(numGroups, -1);
    h_collectionBounds = std::vector<int2>(numCollections, make_int2(-1, -1));
    h_collectionNumActive = std::vector<int>(numCollections, -1);
}

/**
 * Error checking helper routines
 */

void checkAtomIndex(const int numAtoms, const std::string &restType, const int atomIndex,
                    const int restIndex, const int globalIndex, const bool allowNegativeOne = false)
{
    bool bad = false;
    if (allowNegativeOne)
    {
        if (atomIndex < -1)
        {
            bad = true;
        }
    }
    else
    {
        if (atomIndex < 0)
        {
            bad = true;
        }
    }
    if (atomIndex >= numAtoms)
    {
        bad = true;
    }
    if (bad)
    {
        std::stringstream m;
        m << "Bad index given in " << restType << ". atomIndex is " << atomIndex;
        m << ", globalIndex is: " << globalIndex << ", restraint index is: " << restIndex;
        throw OpenMMException(m.str());
    }
}

void checkForceConstant(const float forceConst, const std::string &restType,
                        const int restIndex, const int globalIndex)
{
    if (forceConst < 0)
    {
        std::stringstream m;
        m << "Force constant is < 0 for " << restType << " at globalIndex " << globalIndex << ", restraint index " << restIndex;
        throw OpenMMException(m.str());
    }
}

void checkDistanceRestraintRs(const float r1, const float r2, const float r3,
                              const float r4, const int restIndex, const int globalIndex)
{
    std::stringstream m;
    bool bad = false;
    m << "Distance restraint has ";

    if (r1 > r2)
    {
        m << "r1 > r2. ";
        bad = true;
    }
    else if (r2 > r3)
    {
        m << "r2 > r3. ";
        bad = true;
    }
    else if (r3 > r4)
    {
        m << "r3 > r4. ";
        bad = true;
    }

    if (bad)
    {
        m << "Restraint has index " << restIndex << " and globalIndex " << globalIndex << ".";
        throw OpenMMException(m.str());
    }
}

void checkTorsionRestraintAngles(const float phi, const float deltaPhi, const int index, const int globalIndex)
{
    std::stringstream m;
    bool bad = false;

    if ((phi < -180.) || (phi > 180.))
    {
        m << "Torsion restraint phi lies outside of [-180, 180]. ";
        bad = true;
    }
    if ((deltaPhi < 0) || (deltaPhi > 180))
    {
        m << "Torsion restraint deltaPhi lies outside of [0, 180]. ";
        bad = true;
    }
    if (bad)
    {
        m << "Restraint has index " << index << " and globalIndex " << globalIndex << ".";
        throw OpenMMException(m.str());
    }
}

void checkGroupCollectionIndices(const int num, const std::vector<int> &indices,
                                 std::vector<int> &assigned, const int index,
                                 const std::string &type1, const std::string &type2)
{
    std::stringstream m;
    for (std::vector<int>::const_iterator i = indices.begin(); i != indices.end(); ++i)
    {
        // make sure we're in range
        if ((*i >= num) || (*i < 0))
        {
            m << type2 << " with index " << index << " references " << type1 << " outside of range[0," << (num - 1) << "].";
            throw OpenMMException(m.str());
        }
        // check to see if this restraint is already assigned to another group
        if (assigned[*i] != -1)
        {
            m << type1 << " with index " << (*i) << " is assinged to more than one " << type2 << ". ";
            m << type2 << "s are " << assigned[*i] << " and ";
            m << index << ".";
            throw OpenMMException(m.str());
        }
        // otherwise mark this group as belonging to us
        else
        {
            assigned[*i] = index;
        }
    }
}

void checkNumActive(const std::vector<int> &indices, const int numActive, const int index, const std::string &type)
{
    if ((numActive < 0) || (numActive > indices.size()))
    {
        std::stringstream m;
        m << type << " with index " << index << " has numActive=" << numActive << " out of range [0," << indices.size() << "].";
        throw OpenMMException(m.str());
    }
}

void checkAllAssigned(const std::vector<int> &assigned, const std::string &type1, const std::string &type2)
{
    for (std::vector<int>::const_iterator i = assigned.begin(); i != assigned.end(); ++i)
    {
        if (*i == -1)
        {
            std::stringstream m;
            int index = std::distance(assigned.begin(), i);
            m << type1 << " with index " << index << " is not assigned to a " << type2 << ".";
            throw OpenMMException(m.str());
        }
    }
}

void CudaCalcMeldForceKernel::setupRDCRestraints(const MeldForce &force)
{
    rdcScaleFactor = force.getRDCScaleFactor();
    int numAtoms = system.getNumParticles();
    std::string restType = "RDC restraint";
    for (int i = 0; i < numRDCRestraints; ++i)
    {
        int atom1, atom2, alignment, global_index;
        float kappa, obs, tol, quad_cut, force_constant;
        force.getRDCRestraintParameters(i, atom1, atom2, alignment,
                                        kappa, obs,
                                        tol, quad_cut, force_constant,
                                        global_index);
        checkAtomIndex(numAtoms, restType, atom1, i, global_index, true);
        checkAtomIndex(numAtoms, restType, atom2, i, global_index, true);
        checkForceConstant(force_constant, restType, i, global_index);
        if (alignment < 0)
        {
            throw OpenMMException("Alignment tensor index must be >= 0");
        }
        if (alignment >= force.getNumRDCAlignments())
        {
            throw OpenMMException("Alignment tensor index must be < number of alignment tensors");
        }

        h_rdcRestParams1[i] = make_float2(kappa, obs);
        h_rdcRestParams2[i] = make_float3(tol, quad_cut, force_constant);
        h_rdcRestAtomIndices[i] = make_int2(atom1, atom2);
        h_rdcRestAlignments[i] = alignment;
        h_rdcRestGlobalIndices[i] = global_index;
    }
}

int match_param_name(std::string target, const vector<string> &names)
{
    for (int i = 0; i < names.size(); i++)
    {
        if (names[i] == target)
        {
            return i;
        }
    }
    throw OpenMMException("Could not find parameter derivative index.");
}

void CudaCalcMeldForceKernel::setupRDCDerivIndices()
{
    // Add all of our custom parameters
    for (int i = 0; i < numRDCAlignments; i++)
    {
        std::string base = "rdc_" + std::to_string(i);
        cu.addEnergyParameterDerivative(base + "_s1");
        cu.addEnergyParameterDerivative(base + "_s2");
        cu.addEnergyParameterDerivative(base + "_s3");
        cu.addEnergyParameterDerivative(base + "_s4");
        cu.addEnergyParameterDerivative(base + "_s5");
    }

    // Get all of the parameter names
    const vector<string> &names = cu.getEnergyParamDerivNames();

    // Figure out which parameter corresponds to each name
    int count = 0;
    for (int i = 0; i < numRDCAlignments; i++)
    {
        std::string base = "rdc_" + std::to_string(i);
        for (int j = 1; j < 6; j++)
        {
            std::string suffix = "_s" + std::to_string(j);
            int index = match_param_name(base + suffix, names);
            h_rdcRestDerivIndices[count] = index;
            count += 1;
        }
    }
    if (numRDCAlignments > 0)
    {
        rdcRestDerivIndices->upload(h_rdcRestDerivIndices);
    }
}

void CudaCalcMeldForceKernel::setupDistanceRestraints(const MeldForce &force)
{
    int numAtoms = system.getNumParticles();
    std::string restType = "distance restraint";
    for (int i = 0; i < numDistRestraints; ++i)
    {
        int atom_i, atom_j, global_index;
        float r1, r2, r3, r4, k;
        force.getDistanceRestraintParams(i, atom_i, atom_j, r1, r2, r3, r4, k, global_index);

        checkAtomIndex(numAtoms, restType, atom_i, i, global_index, true);
        checkAtomIndex(numAtoms, restType, atom_j, i, global_index, true);
        checkForceConstant(k, restType, i, global_index);
        checkDistanceRestraintRs(r1, r2, r3, r4, i, global_index);

        h_distanceRestRParams[i] = make_float4(r1, r2, r3, r4);
        h_distanceRestKParams[i] = k;
        h_distanceRestAtomIndices[i] = make_int2(atom_i, atom_j);
        h_distanceRestGlobalIndices[i] = global_index;
    }
}

void CudaCalcMeldForceKernel::setupHyperbolicDistanceRestraints(const MeldForce &force)
{
    int numAtoms = system.getNumParticles();
    std::string restType = "hyperbolic distance restraint";
    for (int i = 0; i < numHyperbolicDistRestraints; ++i)
    {
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

void CudaCalcMeldForceKernel::setupTorsionRestraints(const MeldForce &force)
{
    int numAtoms = system.getNumParticles();
    std::string restType = "torsion restraint";
    for (int i = 0; i < numTorsionRestraints; ++i)
    {
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

void CudaCalcMeldForceKernel::setupDistProfileRestraints(const MeldForce &force)
{
    int numAtoms = system.getNumParticles();
    std::string restType = "distance profile restraint";
    int currentParamIndex = 0;
    for (int i = 0; i < numDistProfileRestraints; ++i)
    {
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

        for (int j = 0; j < nBins; ++j)
        {
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

void CudaCalcMeldForceKernel::setupTorsProfileRestraints(const MeldForce &force)
{
    int numAtoms = system.getNumParticles();
    std::string restType = "torsion profile restraint";
    int currentParamIndex = 0;
    for (int i = 0; i < numTorsProfileRestraints; ++i)
    {
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

        for (int j = 0; j < nBins * nBins; ++j)
        {
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

void CudaCalcMeldForceKernel::setupGMMRestraints(const MeldForce &force)
{
    int atomBlockOffset = 0;
    int dataBlockOffset = 0;

    for (int index = 0; index < force.getNumGMMRestraints(); index++)
    {
        int nPairs, nComponents, globalIndex;
        float scale;
        std::vector<int> atomIndices;
        std::vector<double> weights;
        std::vector<double> means;
        std::vector<double> diag;
        std::vector<double> offdiag;
        force.getGMMRestraintParams(index, nPairs, nComponents, scale, atomIndices,
                                    weights, means, diag, offdiag, globalIndex);
        h_gmmParams[index].x = nPairs;
        h_gmmParams[index].y = nComponents;
        h_gmmParams[index].z = globalIndex;
        h_gmmParams[index].w = (int)(scale * 1e6); // multiple by a million to store in an int
                                                   // divide by a million and convert back to float
                                                   // on gpu

        h_gmmOffsets[index].x = atomBlockOffset;
        h_gmmOffsets[index].y = dataBlockOffset;

        for (int i = 0; i < nPairs; i++)
        {
            h_gmmAtomIndices[atomBlockOffset + 2 * i] = atomIndices[2 * i];
            h_gmmAtomIndices[atomBlockOffset + 2 * i + 1] = atomIndices[2 * i + 1];
        }
        atomBlockOffset += nPairs;

        for (int i = 0; i < nComponents; i++)
        {
            // build the precision matrix
            auto precision = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>(nPairs, nPairs);
            for (int j = 0; j < nPairs; j++)
            {
                precision(j, j) = diag[i * nPairs + j];
            }
            int count = 0;
            for (int j = 0; j < nPairs; ++j)
            {
                for (int k = j + 1; k < nPairs; ++k)
                {
                    precision(j, k) = offdiag[i * nPairs * (nPairs - 1) / 2 + count];
                    precision(k, j) = offdiag[i * nPairs * (nPairs - 1) / 2 + count];
                    count++;
                }
            }

            // compute the eigen values
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> es(precision);
            auto eigenvalues = es.eigenvalues();

            // check that precision is positive definite
            if (eigenvalues.minCoeff() <= 0)
            {
                throw OpenMMException("The precision matrix must be positive definite.");
            }

            // compute determinant
            float det = eigenvalues.prod();

            // compute normalization
            float norm = weights[i] / sqrt(pow(2.0 * 3.141597, nPairs)) * sqrt(det);

            // shove stuff in array
            h_gmmData[dataBlockOffset] = norm;
            for (int j = 0; j < nPairs; j++)
            {
                h_gmmData[dataBlockOffset + j + 1] = means[i * nPairs + j];
                h_gmmData[dataBlockOffset + nPairs + j + 1] = diag[i * nPairs + j];
            }
            for (int j = 0; j < nPairs * (nPairs - 1) / 2; j++)
            {
                h_gmmData[dataBlockOffset + 2 * nPairs + j + 1] = offdiag[i * nPairs * (nPairs - 1) / 2 + j];
            }
            dataBlockOffset += 1 + 2 * nPairs + nPairs * (nPairs - 1) / 2;
        }
    }
}


void CudaCalcMeldForceKernel::setupGridPotentialRestraints(const MeldForce &force)
{
    // the current script supports multiple gridPotentialRest restraints with corresponding density maps,
    // for each gridPotentialRest restraint, it contains the atoms got affected, the grid potential (mu) on density map.
    // gridPotentialRestAtomList is defined to store how many atoms in each gridPotentialRest restraint,
    // e.g. restraint_0 has 2 atoms and restraint_1 has 3 atoms, the gridPotentialRestAtomList will be [0,2,5]
    // all grid potentials for all density maps are stored as a single vector. It will check the atom is in which atom set
    // to determine which density map potential (mu) it should use.
    int numAtoms = system.getNumParticles();
    std::string restType = "density restraint";
    int atomset = 0;
    h_gridPotentialRestAtomList[0] = 0;
    for (int i = 0; i < numGridPotentialRestraints; ++i)
    {
        int global_index;
        std::vector<int> atom;
        std::vector<double> mu,gridpos_x, gridpos_y, gridpos_z;
        force.getGridPotentialRestraintParams(i, atom, mu, gridpos_x, gridpos_y, gridpos_z, global_index);
        for (int d = 0; d < gridpos_x.size(); ++d) {
            h_gridPotentialRestGridPosx[d] = gridpos_x[d];
        }
        for (int d = 0; d < gridpos_y.size(); ++d) {
            h_gridPotentialRestGridPosy[d] = gridpos_y[d];
        }
        for (int d = 0; d < gridpos_z.size(); ++d) {
            h_gridPotentialRestGridPosz[d] = gridpos_z[d];
        }
        for (int d = 0; d < gridpos_x.size()*gridpos_y.size()*gridpos_z.size(); ++d) {
            h_gridPotentialRestMu[i*gridpos_x.size()*gridpos_y.size()*gridpos_z.size()+d] = mu[d];
        }
        for (int a = 0; a < atom.size(); ++a) {
            h_gridPotentialRestWeights[a+atomset] = system.getParticleMass(atom[a]);
            h_gridPotentialRestAtomIndices[a+atomset] = atom[a];
        }
        h_gridPotentialRestGlobalIndices[i] = global_index; 
        atomset+=atom.size();
        h_gridPotentialRestAtomList[i+1] = atomset;
    }
}

void CudaCalcMeldForceKernel::setupGroups(const MeldForce &force)
{
    largestGroup = 0;
    std::vector<int> restraintAssigned(numRestraints, -1);
    int start = 0;
    int end = 0;
    for (int i = 0; i < numGroups; ++i)
    {
        std::vector<int> indices;
        int numActive;
        force.getGroupParams(i, indices, numActive);

        checkGroupCollectionIndices(numRestraints, indices, restraintAssigned, i, "Restraint", "Group");
        checkNumActive(indices, numActive, i, "Group");

        int groupSize = indices.size();
        if (groupSize > largestGroup)
        {
            largestGroup = groupSize;
        }

        end = start + groupSize;
        h_groupNumActive[i] = numActive;
        h_groupBounds[i] = make_int2(start, end);

        for (int j = 0; j < indices.size(); ++j)
        {
            h_groupRestraintIndices[start + j] = indices[j];
        }
        start = end;
    }
    checkAllAssigned(restraintAssigned, "Restraint", "Group");
}

void CudaCalcMeldForceKernel::setupCollections(const MeldForce &force)
{
    largestCollection = 0;
    std::vector<int> groupAssigned(numGroups, -1);
    int start = 0;
    int end = 0;
    for (int i = 0; i < numCollections; ++i)
    {
        std::vector<int> indices;
        int numActive;
        force.getCollectionParams(i, indices, numActive);
        checkGroupCollectionIndices(numGroups, indices, groupAssigned, i, "Group", "Collection");
        checkNumActive(indices, numActive, i, "Collection");

        int collectionSize = indices.size();

        if (collectionSize > largestCollection)
        {
            largestCollection = collectionSize;
        }

        end = start + collectionSize;
        h_collectionNumActive[i] = numActive;
        h_collectionBounds[i] = make_int2(start, end);
        for (int j = 0; j < indices.size(); ++j)
        {
            h_collectionGroupIndices[start + j] = indices[j];
        }
        start = end;
    }
    checkAllAssigned(groupAssigned, "Group", "Collection");
}

int3 CudaCalcMeldForceKernel::calcNumGrids(const MeldForce &force)
{
    int global_index;
    std::vector<int> atom;
    std::vector<double> mu, gridpos_x, gridpos_y, gridpos_z;
    int x_size = 1;
    int y_size = 1;
    int z_size = 1;
    if (numGridPotentialRestraints > 0) {
        force.getGridPotentialRestraintParams(0, atom, mu, gridpos_x, gridpos_y, gridpos_z, global_index);
        x_size = gridpos_x.size();
        y_size = gridpos_y.size();
        z_size = gridpos_z.size();
    }
    return make_int3(x_size,y_size,z_size);
}
int CudaCalcMeldForceKernel::calcNumGridPotentialAtoms(const MeldForce &force)
{
    int total = 0;
    int global_index;
    std::vector<int> atom;
    std::vector<double> mu, gridpos_x, gridpos_y, gridpos_z;
    for (int i=0; i<force.getNumGridPotentialRestraints(); ++i) {
        force.getGridPotentialRestraintParams(i, atom,  mu, gridpos_x, gridpos_y, gridpos_z, global_index);
        total += atom.size();
    }
    return total;
}

int CudaCalcMeldForceKernel::calcSizeGMMAtomIndices(const MeldForce &force)
{
    int total = 0;
    int nPairs;
    int nComponents;
    int globalIndex;
    float scale;
    std::vector<int> atomIndices;
    std::vector<double> weights;
    std::vector<double> means;
    std::vector<double> diags;
    std::vector<double> offdiags;

    for (int i = 0; i < force.getNumGMMRestraints(); ++i)
    {
        force.getGMMRestraintParams(i, nPairs, nComponents, scale,
                                    atomIndices, weights, means,
                                    diags, offdiags, globalIndex);
        total += 2 * nPairs;
    }
    return total;
}

int CudaCalcMeldForceKernel::calcSizeGMMData(const MeldForce &force)
{
    int total = 0;
    int nPairs;
    int nComponents;
    int globalIndex;
    float scale;
    std::vector<int> atomIndices;
    std::vector<double> weights;
    std::vector<double> means;
    std::vector<double> diags;
    std::vector<double> offdiags;

    for (int i = 0; i < force.getNumGMMRestraints(); ++i)
    {
        force.getGMMRestraintParams(i, nPairs, nComponents, scale,
                                    atomIndices, weights, means,
                                    diags, offdiags, globalIndex);
        total +=
            nComponents +                            // weights
            nComponents * nPairs +                   // means
            nComponents * nPairs +                   // precision diagonals
            nComponents * nPairs * (nPairs - 1) / 2; // precision off diagonals
    }
    return total;
}

void CudaCalcMeldForceKernel::validateAndUpload()
{
    if (numRDCRestraints > 0)
    {
        rdcRestParams1->upload(h_rdcRestParams1);
        rdcRestParams2->upload(h_rdcRestParams2);
        rdcRestAtomIndices->upload(h_rdcRestAtomIndices);
        rdcRestGlobalIndices->upload(h_rdcRestGlobalIndices);
        rdcRestAlignmentComponents->upload(h_rdcRestAlignmentComponents);
    }

    if (numRDCAlignments > 0)
    {
        rdcRestAlignments->upload(h_rdcRestAlignments);
        rdcRestDerivIndices->upload(h_rdcRestDerivIndices);
    }

    if (numDistRestraints > 0)
    {
        distanceRestRParams->upload(h_distanceRestRParams);
        distanceRestKParams->upload(h_distanceRestKParams);
        distanceRestAtomIndices->upload(h_distanceRestAtomIndices);
        distanceRestGlobalIndices->upload(h_distanceRestGlobalIndices);
    }

    if (numHyperbolicDistRestraints > 0)
    {
        hyperbolicDistanceRestRParams->upload(h_hyperbolicDistanceRestRParams);
        hyperbolicDistanceRestParams->upload(h_hyperbolicDistanceRestParams);
        hyperbolicDistanceRestAtomIndices->upload(h_hyperbolicDistanceRestAtomIndices);
        hyperbolicDistanceRestGlobalIndices->upload(h_hyperbolicDistanceRestGlobalIndices);
    }

    if (numTorsionRestraints > 0)
    {
        torsionRestParams->upload(h_torsionRestParams);
        torsionRestAtomIndices->upload(h_torsionRestAtomIndices);
        torsionRestGlobalIndices->upload(h_torsionRestGlobalIndices);
    }

    if (numDistProfileRestraints > 0)
    {
        distProfileRestAtomIndices->upload(h_distProfileRestAtomIndices);
        distProfileRestDistRanges->upload(h_distProfileRestDistRanges);
        distProfileRestNumBins->upload(h_distProfileRestNumBins);
        distProfileRestParamBounds->upload(h_distProileRestParamBounds);
        distProfileRestParams->upload(h_distProfileRestParams);
        distProfileRestScaleFactor->upload(h_distProfileRestScaleFactor);
        distProfileRestGlobalIndices->upload(h_distProfileRestGlobalIndices);
    }

    if (numTorsProfileRestraints > 0)
    {
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

    if (numGMMRestraints > 0)
    {
        gmmParams->upload(h_gmmParams);
        gmmOffsets->upload(h_gmmOffsets);
        gmmAtomIndices->upload(h_gmmAtomIndices);
        gmmData->upload(h_gmmData);
    }


    if (numGridPotentialRestraints > 0) {
        gridPotentialRestGridPosx->upload(h_gridPotentialRestGridPosx);
        gridPotentialRestGridPosy->upload(h_gridPotentialRestGridPosy);
        gridPotentialRestGridPosz->upload(h_gridPotentialRestGridPosz);
        gridPotentialRestMu->upload(h_gridPotentialRestMu);
        gridPotentialRestAtomIndices->upload(h_gridPotentialRestAtomIndices);
        gridPotentialRestAtomList->upload(h_gridPotentialRestAtomList);
        gridPotentialRestWeights->upload(h_gridPotentialRestWeights);
        gridPotentialRestGlobalIndices->upload(h_gridPotentialRestGlobalIndices);
    }


    groupRestraintIndices->upload(h_groupRestraintIndices);
    groupBounds->upload(h_groupBounds);
    groupNumActive->upload(h_groupNumActive);
    collectionGroupIndices->upload(h_collectionGroupIndices);
    collectionBounds->upload(h_collectionBounds);
    collectionNumActive->upload(h_collectionNumActive);
}

void CudaCalcMeldForceKernel::initialize(const System &system, const MeldForce &force){
    cu.setAsCurrent();

    allocateMemory(force);
    setupRDCDerivIndices();
    setupRDCRestraints(force);
    setupDistanceRestraints(force);
    setupHyperbolicDistanceRestraints(force);
    setupTorsionRestraints(force);
    setupDistProfileRestraints(force);
    setupTorsProfileRestraints(force);
    setupGMMRestraints(force);
    setupGridPotentialRestraints(force);
    setupGroups(force);
    setupCollections(force);
    validateAndUpload();

    std::map<std::string, std::string> replacements;
    std::map<std::string, std::string> defines;
    defines["NUM_ATOMS"] = cu.intToString(cu.getNumAtoms());
    defines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
    defines["NUM_DERIVS"] = cu.intToString(cu.getEnergyParamDerivNames().size());

    // This should be determined by hardware, rather than hard-coded.
    const int maxThreadsPerGroup = 1024;
    // Note x / y + (x % y !=0) does integer division and round up
    const int restraintsPerThread = std::max(
        4,
        largestGroup / maxThreadsPerGroup + (largestGroup % maxThreadsPerGroup != 0));
    threadsPerGroup = largestGroup / restraintsPerThread + (largestGroup % restraintsPerThread != 0);
    replacements["NGROUPTHREADS"] = cu.intToString(threadsPerGroup);
    replacements["RESTS_PER_THREAD"] = cu.intToString(restraintsPerThread);

    // This should be determined by hardware, rather than hard-coded.
    const int maxThreadsPerCollection = 1024;
    // Note x / y + (x % y !=0) does integer division and round up
    const int groupsPerThread = std::max(
        4,
        largestCollection / maxThreadsPerCollection + (largestCollection % maxThreadsPerCollection != 0));
    threadsPerCollection = largestCollection / groupsPerThread + (largestCollection % groupsPerThread != 0);
    replacements["NCOLLTHREADS"] = cu.intToString(threadsPerCollection);
    replacements["GROUPS_PER_THREAD"] = cu.intToString(groupsPerThread);

    CUmodule module = cu.createModule(cu.replaceStrings(CudaMeldKernelSources::vectorOps + CudaMeldKernelSources::computeMeld, replacements), defines);
    computeRDCRestKernel = cu.getKernel(module, "computeRDCRest");
    computeDistRestKernel = cu.getKernel(module, "computeDistRest");
    computeHyperbolicDistRestKernel = cu.getKernel(module, "computeHyperbolicDistRest");
    computeTorsionRestKernel = cu.getKernel(module, "computeTorsionRest");
    computeDistProfileRestKernel = cu.getKernel(module, "computeDistProfileRest");
    computeTorsProfileRestKernel = cu.getKernel(module, "computeTorsProfileRest");
    computeGMMRestKernel = cu.getKernel(module, "computeGMMRest");
    computeGridPotentialRestKernel = cu.getKernel(module, "computeGridPotentialRest");
    evaluateAndActivateKernel = cu.getKernel(module, "evaluateAndActivate");
    evaluateAndActivateCollectionsKernel = cu.getKernel(module, "evaluateAndActivateCollections");
    applyGroupsKernel = cu.getKernel(module, "applyGroups");
    applyRDCRestKernel = cu.getKernel(module, "applyRDCRest");
    applyDistRestKernel = cu.getKernel(module, "applyDistRest");
    applyHyperbolicDistRestKernel = cu.getKernel(module, "applyHyperbolicDistRest");
    applyTorsionRestKernel = cu.getKernel(module, "applyTorsionRest");
    applyDistProfileRestKernel = cu.getKernel(module, "applyDistProfileRest");
    applyTorsProfileRestKernel = cu.getKernel(module, "applyTorsProfileRest");
    applyGMMRestKernel = cu.getKernel(module, "applyGMMRest");
    applyGridPotentialRestKernel = cu.getKernel(module, "applyGridPotentialRest");
    cu.addForce(new CudaMeldForceInfo(force));   
}

void CudaCalcMeldForceKernel::copyParametersToContext(ContextImpl &context, const MeldForce &force){
    cu.setAsCurrent();

    setupRDCRestraints(force);
    setupDistanceRestraints(force);
    setupHyperbolicDistanceRestraints(force);
    setupTorsionRestraints(force);
    setupDistProfileRestraints(force);
    setupTorsProfileRestraints(force);
    setupGMMRestraints(force);
    setupGridPotentialRestraints(force);
    setupGroups(force);
    setupCollections(force);
    validateAndUpload();
    // Mark that the current reordering may be invalid.
    cu.invalidateMolecules();
}

void CudaCalcMeldForceKernel::updateRDCGlobalParameters(ContextImpl &context)
{
    for (int i = 0; i < numRDCAlignments; i++)
    {
        std::string base = "rdc_" + std::to_string(i);
        float s1 = context.getParameter(base + "_s1");
        float s2 = context.getParameter(base + "_s2");
        float s3 = context.getParameter(base + "_s3");
        float s4 = context.getParameter(base + "_s4");
        float s5 = context.getParameter(base + "_s5");
        h_rdcRestAlignmentComponents[5 * i + 0] = s1;
        h_rdcRestAlignmentComponents[5 * i + 1] = s2;
        h_rdcRestAlignmentComponents[5 * i + 2] = s3;
        h_rdcRestAlignmentComponents[5 * i + 3] = s4;
        h_rdcRestAlignmentComponents[5 * i + 4] = s5;
    }
    rdcRestAlignmentComponents->upload(h_rdcRestAlignmentComponents);
}

double CudaCalcMeldForceKernel::execute(ContextImpl &context, bool includeForces, bool includeEnergy)
{
    // compute the forces and energies
    if (numRDCRestraints > 0)
    {
        updateRDCGlobalParameters(context);
        void *rdcArgs[] = {
            &cu.getPosq().getDevicePointer(),
            &rdcRestAtomIndices->getDevicePointer(),
            &rdcRestParams1->getDevicePointer(),
            &rdcRestParams2->getDevicePointer(),
            &rdcScaleFactor,
            &rdcRestAlignments->getDevicePointer(),
            &rdcRestAlignmentComponents->getDevicePointer(),
            &rdcRestGlobalIndices->getDevicePointer(),
            &restraintEnergies->getDevicePointer(),
            &rdcRestForces->getDevicePointer(),
            &rdcRestDerivs->getDevicePointer(),
            &numRDCRestraints};
        cu.executeKernel(computeRDCRestKernel, rdcArgs, numRDCRestraints);
    }

    if (numDistRestraints > 0)
    {
        void *distanceArgs[] = {
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

    if (numHyperbolicDistRestraints > 0)
    {
        void *hyperbolicDistanceArgs[] = {
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

    if (numTorsionRestraints > 0)
    {
        void *torsionArgs[] = {
            &cu.getPosq().getDevicePointer(),
            &torsionRestAtomIndices->getDevicePointer(),
            &torsionRestParams->getDevicePointer(),
            &torsionRestGlobalIndices->getDevicePointer(),
            &restraintEnergies->getDevicePointer(),
            &torsionRestForces->getDevicePointer(),
            &numTorsionRestraints};
        cu.executeKernel(computeTorsionRestKernel, torsionArgs, numTorsionRestraints);
    }

    if (numDistProfileRestraints > 0)
    {
        void *distProfileArgs[] = {
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
            &numDistProfileRestraints};
        cu.executeKernel(computeDistProfileRestKernel, distProfileArgs, numDistProfileRestraints);
    }

    if (numTorsProfileRestraints > 0)
    {
        void *torsProfileArgs[] = {
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
            &numTorsProfileRestraints};
        cu.executeKernel(computeTorsProfileRestKernel, torsProfileArgs, numTorsProfileRestraints);
    }

    if (numGMMRestraints > 0)
    {
        void *gmmArgs[] = {
            &cu.getPosq().getDevicePointer(),
            &numGMMRestraints,
            &gmmParams->getDevicePointer(),
            &gmmOffsets->getDevicePointer(),
            &gmmAtomIndices->getDevicePointer(),
            &gmmData->getDevicePointer(),
            &restraintEnergies->getDevicePointer(),
            &gmmForces->getDevicePointer()};
        cu.executeKernel(computeGMMRestKernel, gmmArgs, numGMMRestraints * 32, 32 * 16, 2 * 16 * 32 * sizeof(float));
    }

    if (numGridPotentialRestraints > 0) 
    {
        void* gridPotentialRestArgs[] = {
            &cu.getPosq().getDevicePointer(),
            &gridPotentialRestAtomIndices->getDevicePointer(),
            &gridPotentialRestGridPosx->getDevicePointer(),
            &gridPotentialRestGridPosy->getDevicePointer(),
            &gridPotentialRestGridPosz->getDevicePointer(),
            &gridPotentialRestMu->getDevicePointer(),
            &gridPotentialRestWeights->getDevicePointer(),
            &gridPotentialRestAtomList->getDevicePointer(),
            &gridPotentialRestGlobalIndices->getDevicePointer(),
            &restraintEnergies->getDevicePointer(),
            &gridPotentialRestForces->getDevicePointer(),
            &numGridPotentialRestraints,
            &numGridPotentialGrids,
            &numGridPotentialAtoms};
        cu.executeKernel(computeGridPotentialRestKernel, gridPotentialRestArgs, numGridPotentialAtoms);
    }

    // now evaluate and activate restraints based on groups
    void *groupArgs[] = {
        &numGroups,
        &groupNumActive->getDevicePointer(),
        &groupBounds->getDevicePointer(),
        &groupRestraintIndices->getDevicePointer(),
        &restraintEnergies->getDevicePointer(),
        &restraintActive->getDevicePointer(),
        &groupEnergies->getDevicePointer(),
    };
    cu.executeKernel(evaluateAndActivateKernel, groupArgs, threadsPerGroup * numGroups, threadsPerGroup);

    // now evaluate and activate groups based on collections
    void *collArgs[] = {
        &numCollections,
        &collectionNumActive->getDevicePointer(),
        &collectionBounds->getDevicePointer(),
        &collectionGroupIndices->getDevicePointer(),
        &groupEnergies->getDevicePointer(),
        &groupActive->getDevicePointer()};
    cu.executeKernel(evaluateAndActivateCollectionsKernel, collArgs, threadsPerCollection * numCollections, threadsPerCollection);

    // Now set the restraints active based on if the groups are active
    void *applyGroupsArgs[] = {
        &groupActive->getDevicePointer(),
        &restraintActive->getDevicePointer(),
        &groupBounds->getDevicePointer(),
        &numGroups};
    cu.executeKernel(applyGroupsKernel, applyGroupsArgs, 32 * numGroups, 32);

    // Now apply the forces and energies if the restraints are active
    if (numRDCRestraints > 0)
    {
        void *applyRDCArgs[] = {
            &cu.getForce().getDevicePointer(),
            &cu.getEnergyBuffer().getDevicePointer(),
            &cu.getEnergyParamDerivBuffer().getDevicePointer(),
            &rdcRestAtomIndices->getDevicePointer(),
            &rdcRestAlignments->getDevicePointer(),
            &rdcRestGlobalIndices->getDevicePointer(),
            &rdcRestForces->getDevicePointer(),
            &rdcRestDerivs->getDevicePointer(),
            &rdcRestDerivIndices->getDevicePointer(),
            &restraintEnergies->getDevicePointer(),
            &restraintActive->getDevicePointer(),
            &numRDCRestraints};
        cu.executeKernel(applyRDCRestKernel, applyRDCArgs, numRDCRestraints);
    }

    if (numDistRestraints > 0)
    {
        void *applyDistRestArgs[] = {
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

    if (numHyperbolicDistRestraints > 0)
    {
        void *applyHyperbolicDistRestArgs[] = {
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

    if (numTorsionRestraints > 0)
    {
        void *applyTorsionRestArgs[] = {
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

    if (numDistProfileRestraints > 0)
    {
        void *applyDistProfileRestArgs[] = {
            &cu.getForce().getDevicePointer(),
            &cu.getEnergyBuffer().getDevicePointer(),
            &distProfileRestAtomIndices->getDevicePointer(),
            &distProfileRestGlobalIndices->getDevicePointer(),
            &distProfileRestForces->getDevicePointer(),
            &restraintEnergies->getDevicePointer(),
            &restraintActive->getDevicePointer(),
            &numDistProfileRestraints};
        cu.executeKernel(applyDistProfileRestKernel, applyDistProfileRestArgs, numDistProfileRestraints);
    }

    if (numTorsProfileRestraints > 0)
    {
        void *applyTorsProfileRestArgs[] = {
            &cu.getForce().getDevicePointer(),
            &cu.getEnergyBuffer().getDevicePointer(),
            &torsProfileRestAtomIndices0->getDevicePointer(),
            &torsProfileRestAtomIndices1->getDevicePointer(),
            &torsProfileRestGlobalIndices->getDevicePointer(),
            &torsProfileRestForces->getDevicePointer(),
            &restraintEnergies->getDevicePointer(),
            &restraintActive->getDevicePointer(),
            &numTorsProfileRestraints};
        cu.executeKernel(applyTorsProfileRestKernel, applyTorsProfileRestArgs, numTorsProfileRestraints);
    }

    if (numGMMRestraints > 0)
    {
        void *applyGMMRestArgs[] = {
            &cu.getForce().getDevicePointer(),
            &cu.getEnergyBuffer().getDevicePointer(),
            &numGMMRestraints,
            &gmmParams->getDevicePointer(),
            &restraintEnergies->getDevicePointer(),
            &restraintActive->getDevicePointer(),
            &gmmOffsets->getDevicePointer(),
            &gmmAtomIndices->getDevicePointer(),
            &gmmForces->getDevicePointer()};
        cu.executeKernel(applyGMMRestKernel, applyGMMRestArgs, 32 * numGMMRestraints);
    }

    if (numGridPotentialRestraints > 0) {
        void *applyGridPotentialRestArgs[] = {
            &cu.getForce().getDevicePointer(),
            &cu.getEnergyBuffer().getDevicePointer(),
            &gridPotentialRestAtomIndices->getDevicePointer(),
            &gridPotentialRestAtomList->getDevicePointer(),
            &gridPotentialRestGlobalIndices->getDevicePointer(),
            &restraintEnergies->getDevicePointer(),
            &restraintActive->getDevicePointer(),
            &gridPotentialRestForces->getDevicePointer(),
            &numGridPotentialRestraints,
            &numGridPotentialAtoms};
        cu.executeKernel(applyGridPotentialRestKernel, applyGridPotentialRestArgs, numGridPotentialAtoms);
    };

    return 0.0;
}
