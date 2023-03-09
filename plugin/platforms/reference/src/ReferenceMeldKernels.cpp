#include "ReferenceMeldKernels.h"
#include "MeldForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/RealVec.h"
#include "openmm/reference/ReferencePlatform.h"
#include <string>
#include <sstream>
#include <iostream>
#include <Eigen/Dense>
#include "MeldVecTypes.h"
#include "ComputeForces.h"

using namespace MeldPlugin;
using namespace OpenMM;
using namespace std;

static vector<RealVec> &extractPositions(ContextImpl &context)
{
    ReferencePlatform::PlatformData *data = reinterpret_cast<ReferencePlatform::PlatformData *>(context.getPlatformData());
    return *((vector<RealVec> *)data->positions);
}

static vector<RealVec> &extractForces(ContextImpl &context)
{
    ReferencePlatform::PlatformData *data = reinterpret_cast<ReferencePlatform::PlatformData *>(context.getPlatformData());
    return *((vector<RealVec> *)data->forces);
}

void checkAtomIndex(const int numAtoms, const std::string &restType, const int atomIndex,
                    const int restIndex, const int globalIndex, const bool allowNegativeOne = false )
{
    bool bad = false;
    if (allowNegativeOne) {
        if (atomIndex < -1) {
            bad = true;
        }
    } else {
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
        m << type << " with index " << index << " has numActive out of range [0," << indices.size() << "].";
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

ReferenceCalcMeldForceKernel::ReferenceCalcMeldForceKernel(std::string name, const OpenMM::Platform &platform,
                                                           const System &system) : CalcMeldForceKernel(name, platform), system(system)
{
    numRDCRestraints = 0;
    numRDCAlignments = 0;
    numDistRestraints = 0;
    numHyperbolicDistRestraints = 0;
    numTorsionRestraints = 0;
    numDistProfileRestraints = 0;
    numGMMRestraints = 0;
    numGridPotentials = 0;
    numGridPotentialRestraints = 0;
    numRestraints = 0;
    numGroups = 0;
    numCollections = 0;
    rdcScaleFactor = 0.0;
}

void ReferenceCalcMeldForceKernel::initialize(const System &system, const MeldForce &force)
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
    numRestraints = force.getNumTotalRestraints();
    numGroups = force.getNumGroups();
    numCollections = force.getNumCollections();

    rdcRestParams1 = std::vector<float2>(numRDCRestraints, float2(0, 0));
    rdcRestParams2 = std::vector<float3>(numRDCRestraints, float3(0, 0, 0));
    rdcRestAlignments = std::vector<int>(numRDCRestraints, -1);
    rdcRestAtomIndices = std::vector<int2>(numRDCRestraints, int2(-1, -1));
    rdcRestGlobalIndices = std::vector<int>(numRDCRestraints, -1);
    rdcRestForces = std::vector<float3>(numRDCRestraints, float3(0, 0, 0));
    rdcRestAlignmentComponents = std::vector<float>(5 * numRDCAlignments, 0.0);
    rdcRestDerivs = std::vector<float>(5 * numRDCRestraints, 0.0);
    rdcScaleFactor = force.getRDCScaleFactor();

    distanceRestRParams = std::vector<float4>(numDistRestraints, float4(0, 0, 0, 0));
    distanceRestKParams = std::vector<float>(numDistRestraints, 0);
    distanceRestAtomIndices = std::vector<int2>(numDistRestraints, int2(-1, -1));
    distanceRestGlobalIndices = std::vector<int>(numDistRestraints, -1);
    distanceRestForces = std::vector<float3>(numDistRestraints, float3(0, 0, 0));

    hyperbolicDistanceRestRParams = std::vector<float4>(numHyperbolicDistRestraints, float4(0, 0, 0, 0));
    hyperbolicDistanceRestParams = std::vector<float4>(numHyperbolicDistRestraints, float4(0, 0, 0, 0));
    hyperbolicDistanceRestAtomIndices = std::vector<int2>(numHyperbolicDistRestraints, int2(-1, -1));
    hyperbolicDistanceRestGlobalIndices = std::vector<int>(numHyperbolicDistRestraints, -1);

    torsionRestParams = std::vector<float3>(numTorsionRestraints, float3(0, 0, 0));
    torsionRestAtomIndices = std::vector<int4>(numTorsionRestraints, int4(-1, -1, -1, -1));
    torsionRestGlobalIndices = std::vector<int>(numTorsionRestraints, -1);
    torsionRestForces = std::vector<float3>(4 * numTorsionRestraints, float3(0, 0, 0));
    
    distProfileRestAtomIndices = std::vector<int2>(numDistProfileRestraints, int2(-1, -1));
    distProfileRestDistRanges = std::vector<float2>(numDistProfileRestraints, float2(0, 0));
    distProfileRestNumBins = std::vector<int>(numDistProfileRestraints, -1);
    distProfileRestParamBounds = std::vector<int2>(numDistProfileRestraints, int2(-1, -1));
    distProfileRestParams = std::vector<float4>(numDistProfileRestParams, float4(0, 0, 0, 0));
    distProfileRestScaleFactor = std::vector<float>(numDistProfileRestraints, 0);
    distProfileRestGlobalIndices = std::vector<int>(numDistProfileRestraints, -1);

    torsProfileRestAtomIndices0 = std::vector<int4>(numTorsProfileRestraints, int4(-1, -1, -1, -1));
    torsProfileRestAtomIndices1 = std::vector<int4>(numTorsProfileRestraints, int4(-1, -1, -1, -1));
    torsProfileRestNumBins = std::vector<int>(numTorsProfileRestraints, -1);
    torsProfileRestParamBounds = std::vector<int2>(numTorsProfileRestraints, int2(-1, -1));
    torsProfileRestParams0 = std::vector<float4>(numTorsProfileRestParams, float4(0, 0, 0, 0));
    torsProfileRestParams1 = std::vector<float4>(numTorsProfileRestParams, float4(0, 0, 0, 0));
    torsProfileRestParams2 = std::vector<float4>(numTorsProfileRestParams, float4(0, 0, 0, 0));
    torsProfileRestParams3 = std::vector<float4>(numTorsProfileRestParams, float4(0, 0, 0, 0));
    torsProfileRestScaleFactor = std::vector<float>(numTorsProfileRestraints, 0);
    torsProfileRestGlobalIndices = std::vector<int>(numTorsProfileRestraints, -1);

    gmmParams = std::vector<int4>(numGMMRestraints, int4(0, 0, 0, 0));
    gmmOffsets = std::vector<int2>(numGMMRestraints, int2(0, 0));
    gmmAtomIndices = std::vector<int>(calcSizeGMMAtomIndices(force), 0);
    gmmData = std::vector<float>(calcSizeGMMData(force), 0);

    gridPotentials = std::vector<float>(numGridPotentials * get<0>(calcNumGrids(force)) * get<1>(calcNumGrids(force)) * get<2>(calcNumGrids(force)),0);
    gridPotentialgridx = std::vector<float>(get<0>(calcNumGrids(force)), 0);
    gridPotentialgridy = std::vector<float>(get<1>(calcNumGrids(force)), 0);
    gridPotentialgridz = std::vector<float>(get<2>(calcNumGrids(force)), 0);   
    gridPotentialnxyz = std::vector<int>(3, -1);
    gridPotentialRestAtomIndices = std::vector<int>(numGridPotentialRestraints, -1);
    gridPotentialRestWeights = std::vector<float>(numGridPotentialRestraints, 0);
    gridPotentialRestGridPotentoalIndices = std::vector<int>(numGridPotentialRestraints, -1);
    gridPotentialRestGlobalIndices = std::vector<int>(numGridPotentialRestraints, -1);
    gridPotentialRestForces = std::vector<float3>(numGridPotentialRestraints, float3(0, 0, 0));
    restraintEnergies = std::vector<float>(numRestraints, 0);
    restraintActive = std::vector<bool>(numRestraints, false);

    groupRestraintIndices = std::vector<int>(numRestraints, -1);
    groupBounds = std::vector<int2>(numGroups, int2(-1, -1));
    groupNumActive = std::vector<int>(numGroups, -1);
    groupEnergies = std::vector<float>(numGroups, 0);
    groupActive = std::vector<bool>(numGroups, false);

    collectionGroupIndices = std::vector<int>(numGroups, -1);
    collectionBounds = std::vector<int2>(numCollections, int2(-1, -1));
    collectionNumActive = std::vector<int>(numCollections, -1);

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
}

void ReferenceCalcMeldForceKernel::updateRDCGlobalParameters(ContextImpl& context)
{
    for (int i = 0; i < numRDCAlignments; i++)
    {
        std::string base = "rdc_" + std::to_string(i);
        float s1 = context.getParameter(base + "_s1");
        float s2 = context.getParameter(base + "_s2");
        float s3 = context.getParameter(base + "_s3");
        float s4 = context.getParameter(base + "_s4");
        float s5 = context.getParameter(base + "_s5");
        rdcRestAlignmentComponents[5 * i + 0] = s1;
        rdcRestAlignmentComponents[5 * i + 1] = s2;
        rdcRestAlignmentComponents[5 * i + 2] = s3;
        rdcRestAlignmentComponents[5 * i + 3] = s4;
        rdcRestAlignmentComponents[5 * i + 4] = s5;
    }
}

double ReferenceCalcMeldForceKernel::execute(ContextImpl &context, bool includeForces, bool includeEnergy)
{
    vector<RealVec> &pos = extractPositions(context);
    vector<RealVec> &force = extractForces(context);

    if (numRDCRestraints > 0)
    {
        fill(rdcRestForces.begin(), rdcRestForces.end(), float3(0, 0, 0));
        updateRDCGlobalParameters(context);
        computeRDCRest(
            pos,
            rdcRestAtomIndices,
            rdcRestParams1,
            rdcRestParams2,
            rdcScaleFactor,
            rdcRestAlignments,
            rdcRestAlignmentComponents,
            rdcRestGlobalIndices,
            restraintEnergies,
            rdcRestForces,
            rdcRestDerivs,
            numRDCRestraints
        );
    }

    if (numDistRestraints > 0)
    {
        fill(distanceRestForces.begin(), distanceRestForces.end(), float3(0, 0, 0));
        computeDistRest(
            pos,
            distanceRestAtomIndices,
            distanceRestRParams,
            distanceRestKParams,
            distanceRestGlobalIndices,
            restraintEnergies,
            distanceRestForces,
            numDistRestraints);
    }

    if (numHyperbolicDistRestraints > 0)
    {
        fill(hyperbolicDistanceRestForces.begin(), hyperbolicDistanceRestForces.end(), float3(0, 0, 0));
        computeHyperbolicDistRest(
            pos,
            hyperbolicDistanceRestAtomIndices,
            hyperbolicDistanceRestRParams,
            hyperbolicDistanceRestParams,
            hyperbolicDistanceRestGlobalIndices,
            restraintEnergies,
            hyperbolicDistanceRestForces,
            numHyperbolicDistRestraints);
    }

    if (numTorsionRestraints > 0)
    {
        fill(torsionRestForces.begin(), torsionRestForces.end(), float3(0, 0, 0));
        computeTorsionRest(
            pos,
            torsionRestAtomIndices,
            torsionRestParams,
            torsionRestGlobalIndices,
            restraintEnergies,
            torsionRestForces,
            numTorsionRestraints);
    }

    if (numDistProfileRestraints > 0)
    {
        fill(distProfileRestForces.begin(), distProfileRestForces.end(), float3(0, 0, 0));
        computeDistProfileRest(
            pos,
            distProfileRestAtomIndices,
            distProfileRestDistRanges,
            distProfileRestNumBins,
            distProfileRestParams,
            distProfileRestParamBounds,
            distProfileRestScaleFactor,
            distProfileRestGlobalIndices,
            restraintEnergies,
            distProfileRestForces,
            numDistProfileRestraints);
    }

    if (numTorsProfileRestraints > 0)
    {
        fill(torsProfileRestForces.begin(), torsProfileRestForces.end(), float3(0, 0, 0));
        computeTorsProfileRest(
            pos,
            torsProfileRestAtomIndices0,
            torsProfileRestAtomIndices1,
            torsProfileRestNumBins,
            torsProfileRestParams0,
            torsProfileRestParams1,
            torsProfileRestParams2,
            torsProfileRestParams3,
            torsProfileRestParamBounds,
            torsProfileRestScaleFactor,
            torsProfileRestGlobalIndices,
            restraintEnergies,
            torsProfileRestForces,
            numTorsProfileRestraints);
    }

    if (numGMMRestraints > 0)
    {
        fill(gmmForces.begin(), gmmForces.end(), float3(0, 0, 0));
        computeGMMRest(
            pos,
            numGMMRestraints,
            gmmParams,
            gmmOffsets,
            gmmAtomIndices,
            gmmData,
            restraintEnergies,
            gmmForces);
    }

    // if (numGridPotentialRestraints > 0)
    // {
    //     fill(gmmForces.begin(), gmmForces.end(), float3(0, 0, 0));
    //     computeGridPotentialRest(
    //         pos,
    //         gridPotentialRestAtomIndices, 
    //         gridPotentials,
    //         gridPotentialgridx,
    //         gridPotentialgridy,
    //         gridPotentialgridz,
    //         gridPotentialRestWeights,
    //         gridPotentialnxyz,
    //         gridPotentialRestGridPotentoalIndices,
    //         gridPotentialRestGlobalIndices,
    //         numGridPotentialRestraints,
    //         restraintEnergies,    
    //         gridPotentialRestForces);
    // }

    // now evaluate and active restraints based on groups
    evaluateAndActivate(
        numGroups,
        groupNumActive,
        groupBounds,
        groupRestraintIndices,
        restraintEnergies,
        restraintActive,
        groupEnergies);

    // now evaluate and activate groups based on collections
    evaluateAndActivateCollections(
        numCollections,
        collectionNumActive,
        collectionBounds,
        collectionGroupIndices,
        groupEnergies,
        groupActive);

    // Now set the restraints active based on if the groups are active
    applyGroups(
        groupActive,
        restraintActive,
        groupBounds,
        numGroups);


    // Now apply the forces and energies if the restraints are active
    float energy = 0.0;
    if (numRDCRestraints > 0)
    {
        energy += applyRDCRest(
            force,
            rdcRestAtomIndices,
            rdcRestAlignments,
            rdcRestGlobalIndices,
            rdcRestForces,
            rdcRestDerivs,
            restraintEnergies,
            restraintActive,
            extractEnergyParameterDerivatives(context),
            numRDCRestraints);
    }

    if (numDistRestraints > 0)
    {
        energy += applyDistRest(
            force,
            distanceRestAtomIndices,
            distanceRestGlobalIndices,
            distanceRestForces,
            restraintEnergies,
            restraintActive,
            numDistRestraints);
    }

    if (numHyperbolicDistRestraints > 0)
    {
        energy += applyHyperbolicDistRest(
            force,
            hyperbolicDistanceRestAtomIndices,
            hyperbolicDistanceRestGlobalIndices,
            hyperbolicDistanceRestForces,
            restraintEnergies,
            restraintActive,
            numHyperbolicDistRestraints);
    }

    if (numTorsionRestraints > 0)
    {
        energy += applyTorsionRest(
            force,
            torsionRestAtomIndices,
            torsionRestGlobalIndices,
            torsionRestForces,
            restraintEnergies,
            restraintActive,
            numTorsionRestraints);
    }

    if (numDistProfileRestraints > 0)
    {
        energy += applyDistProfileRest(
            force,
            distProfileRestAtomIndices,
            distProfileRestGlobalIndices,
            distProfileRestForces,
            restraintEnergies,
            restraintActive,
            numDistProfileRestraints);
    }

    if (numTorsProfileRestraints > 0)
    {
        energy += applyTorsProfileRest(
            force,
            torsProfileRestAtomIndices0,
            torsProfileRestAtomIndices1,
            torsProfileRestGlobalIndices,
            torsProfileRestForces,
            restraintEnergies,
            restraintActive,
            numTorsProfileRestraints);
    }

    if (numGMMRestraints > 0)
    {
        energy += applyGMMRest(
            force,
            numGMMRestraints,
            gmmParams,
            restraintEnergies,
            restraintActive,
            gmmOffsets,
            gmmAtomIndices,
            gmmForces);
    }

    // if (numGridPotentialRestraints > 0)
    // {
    //     energy += applyGridPotentialRest(
    //         force,
    //         gridPotentialRestAtomIndices,
    //         gridPotentialRestGlobalIndices,
    //         restraintEnergies,
    //         restraintActive,
    //         gridPotentialRestForces,
    //         numGridPotentialRestraints);
    // }    
    return energy;
}

void ReferenceCalcMeldForceKernel::copyParametersToContext(ContextImpl &context, const MeldForce &force)
{
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
}

int ReferenceCalcMeldForceKernel::calcSizeGMMAtomIndices(const MeldForce &force)
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

int ReferenceCalcMeldForceKernel::calcSizeGMMData(const MeldForce &force)
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

int3 ReferenceCalcMeldForceKernel::calcNumGrids(const MeldForce &force)
{
    int nx, ny, nz;
    // float originx, originy, originz, gridx, gridy, gridz;
    // std::vector<double> potential;
    // if (numGridPotentialRestraints > 0) {
    //     force.getGridPotentialParams(0, potential, originx, originy, originz, gridx, gridy, gridz, nx, ny, nz);
    // }
    return int3(nx,ny,nz);
}

void ReferenceCalcMeldForceKernel::setupRDCRestraints(const MeldForce& force)
{
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

        rdcRestParams1[i] = float2(kappa, obs);
        rdcRestParams2[i] = float3(tol, quad_cut, force_constant);
        rdcRestAtomIndices[i] = int2(atom1, atom2);
        rdcRestAlignments[i] = alignment;
        rdcRestGlobalIndices[i] = global_index;
    }
}

void ReferenceCalcMeldForceKernel::setupDistanceRestraints(const MeldForce& force)
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

        distanceRestRParams[i] = float4(r1, r2, r3, r4);
        distanceRestKParams[i] = k;
        distanceRestAtomIndices[i] = int2(atom_i, atom_j);
        distanceRestGlobalIndices[i] = global_index;
    }
}

void ReferenceCalcMeldForceKernel::setupHyperbolicDistanceRestraints(const MeldForce &force)
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

        hyperbolicDistanceRestRParams[i] = float4(r1, r2, r3, r4);
        hyperbolicDistanceRestParams[i] = float4(k1, k2, a, b);
        hyperbolicDistanceRestAtomIndices[i] = int2(atom_i, atom_j);
        hyperbolicDistanceRestGlobalIndices[i] = global_index;
    }
}

void ReferenceCalcMeldForceKernel::setupTorsionRestraints(const MeldForce &force)
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

        torsionRestParams[i] = float3(phi, deltaPhi, forceConstant);
        torsionRestAtomIndices[i] = int4(atom_i, atom_j, atom_k, atom_l);
        torsionRestGlobalIndices[i] = globalIndex;
    }
}

void ReferenceCalcMeldForceKernel::setupDistProfileRestraints(const MeldForce &force)
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

        distProfileRestAtomIndices[i] = int2(atom1, atom2);
        distProfileRestDistRanges[i] = float2(rMin, rMax);
        distProfileRestNumBins[i] = nBins;
        distProfileRestGlobalIndices[i] = globalIndex;
        distProfileRestScaleFactor[i] = scaleFactor;

        for (int j = 0; j < nBins; ++j)
        {
            distProfileRestParams[currentParamIndex] = float4(
                (float)a0[j],
                (float)a1[j],
                (float)a2[j],
                (float)a3[j]);
            currentParamIndex++;
        }
        int thisEnd = currentParamIndex;
        distProfileRestParamBounds[i] = int2(thisStart, thisEnd);
    }
}

void ReferenceCalcMeldForceKernel::setupTorsProfileRestraints(const MeldForce &force)
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

        torsProfileRestAtomIndices0[i] = int4(atom1, atom2, atom3, atom4);
        torsProfileRestAtomIndices1[i] = int4(atom5, atom6, atom7, atom8);
        torsProfileRestNumBins[i] = nBins;
        torsProfileRestGlobalIndices[i] = globalIndex;
        torsProfileRestScaleFactor[i] = scaleFactor;

        for (int j = 0; j < nBins * nBins; ++j)
        {
            torsProfileRestParams0[currentParamIndex] = float4(
                (float)a0[j],
                (float)a1[j],
                (float)a2[j],
                (float)a3[j]);
            torsProfileRestParams1[currentParamIndex] = float4(
                (float)a4[j],
                (float)a5[j],
                (float)a6[j],
                (float)a7[j]);
            torsProfileRestParams2[currentParamIndex] = float4(
                (float)a8[j],
                (float)a9[j],
                (float)a10[j],
                (float)a11[j]);
            torsProfileRestParams3[currentParamIndex] = float4(
                (float)a12[j],
                (float)a13[j],
                (float)a14[j],
                (float)a15[j]);
            currentParamIndex++;
        }
        int thisEnd = currentParamIndex;
        torsProfileRestParamBounds[i] = int2(thisStart, thisEnd);
    }
}

void ReferenceCalcMeldForceKernel::setupGMMRestraints(const MeldForce &force)
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
        gmmParams[index] = int4(nPairs, nComponents, globalIndex, scale);
        gmmOffsets[index] = int2(atomBlockOffset, dataBlockOffset);

        for (int i = 0; i < nPairs; i++)
        {
            gmmAtomIndices[atomBlockOffset + 2 * i] = atomIndices[2 * i];
            gmmAtomIndices[atomBlockOffset + 2 * i + 1] = atomIndices[2 * i + 1];
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
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> > es(precision);
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
            gmmData[dataBlockOffset] = norm;
            for (int j = 0; j < nPairs; j++)
            {
                gmmData[dataBlockOffset + j + 1] = means[i * nPairs + j];
                gmmData[dataBlockOffset + nPairs + j + 1] = diag[i * nPairs + j];
            }
            for (int j = 0; j < nPairs * (nPairs - 1) / 2; j++)
            {
                gmmData[dataBlockOffset + 2 * nPairs + j + 1] = offdiag[i * nPairs * (nPairs - 1) / 2 + j];
            }
            dataBlockOffset += 1 + 2 * nPairs + nPairs * (nPairs - 1) / 2;
        }
    }
}

void ReferenceCalcMeldForceKernel::setupGridPotentialRestraints(const MeldForce &force)
{
//     int numAtoms = system.getNumParticles();
//     std::string restType = "density restraint";
//     for (int d = 0; d < numGridPotentials; ++d) {
//         int nx,ny,nz;
//         float originx, originy, originz, gridx, gridy, gridz;
//         std::vector<double> potential;
//         force.getGridPotentialParams(d, potential,originx,originy,originz,
//                                     gridx,gridy,gridz,nx,ny,nz);
//         for (int i = 0; i < nx; ++i) {
//             gridPotentialgridx[i] = originx+i*gridx;
//         }
//         for (int i = 0; i < ny; ++i) {
//             gridPotentialgridy[i] = originy+i*gridy;
//         }
//         for (int i = 0; i < nz; ++i) {
//             gridPotentialgridz[i] = originz+i*gridz;
//         }
//         for (int i = 0; i < nx*ny*nz; ++i) {
//             gridPotentials[d*nx*ny*nz+i] = potential[i];
//         }
//         gridPotentialnxyz[0] = nx;
//         gridPotentialnxyz[1] = ny;
//         gridPotentialnxyz[2] = nz;
//     }

//     for (int i = 0; i < numGridPotentialRestraints; ++i)
//     {
//         int particle, global_index, potentialGridIndex;
//         float strength;
//         force.getGridPotentialRestraintParams(i, particle, potentialGridIndex, strength, global_index);
//         gridPotentialRestAtomIndices[i] = particle;
//         gridPotentialRestGridPotentoalIndices[i] = potentialGridIndex;
//         gridPotentialRestGlobalIndices[i] = global_index;
//         gridPotentialRestWeights[i] = system.getParticleMass(particle);
//     }
}

void ReferenceCalcMeldForceKernel::setupGroups(const MeldForce &force)
{
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
        end = start + groupSize;
        groupNumActive[i] = numActive;
        groupBounds[i] = int2(start, end);

        for (int j = 0; j < indices.size(); ++j)
        {
            groupRestraintIndices[start + j] = indices[j];
        }
        start = end;
    }
    checkAllAssigned(restraintAssigned, "Restraint", "Group");
}

void ReferenceCalcMeldForceKernel::setupCollections(const MeldForce &force)
{
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
        end = start + collectionSize;
        collectionNumActive[i] = numActive;
        collectionBounds[i] = int2(start, end);
        for (int j = 0; j < indices.size(); ++j)
        {
            collectionGroupIndices[start + j] = indices[j];
        }
        start = end;
    }
    checkAllAssigned(groupAssigned, "Group", "Collection");
}

map<string, double>& ReferenceCalcMeldForceKernel::extractEnergyParameterDerivatives(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *data->energyParameterDerivatives;
}
