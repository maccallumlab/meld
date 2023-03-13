#include "openmm/reference/RealVec.h"
#include "MeldVecTypes.h"

using namespace OpenMM;
using namespace std;

void computeRDCRest(
    vector<RealVec>& pos,
    vector<int2>& rdcRestAtomIndices,
    vector<float2>& rdcRestParams1,
    vector<float3>& rdcRestParams2,
    float scaleFactor,
    vector<int>& rdcRestAlignments,
    vector<float>& rdcRestTensorComponents,
    vector<int>& rdcRestGlobalIndices,
    vector<float>& restraintEnergies,
    vector<float3>& rdcRestForces,
    vector<float>& rdcRestDerivs,
    int numRDCRestraints
);

void computeDistRest(
    vector<RealVec> &pos,
    vector<int2> &distanceRestAtomIndices,
    vector<float4> &distanceRestRParams,
    vector<float> &distanceRestKParams,
    vector<int> &distanceRestGlobalIndices,
    vector<float> &restraintEnergies,
    vector<float3> &distanceRestForces,
    int numDistRestraints);

void computeHyperbolicDistRest(
    vector<RealVec> &pos,
    vector<int2> &hyperbolicDistanceRestAtomIndices,
    vector<float4> &hyperbolicDistanceRestRParams,
    vector<float4> &hyperbolicDistanceRestParams,
    vector<int> &hyperbolicDistanceRestGlobalIndices,
    vector<float> &restraintEnergies,
    vector<float3> &hyperbolicDistanceRestForces,
    int numHyperbolicDistRestraints);

void computeTorsionRest(
    vector<RealVec> &pos,
    vector<int4> &torsionRestAtomIndices,
    vector<float3> &torsionRestParams,
    vector<int> &torsionRestGlobalIndices,
    vector<float> &restraintEnergies,
    vector<float3> &torsionRestForces,
    int numTorsionRestraints);

void computeDistProfileRest(
    vector<RealVec> &pos,
    vector<int2> &distProfileRestAtomIndices,
    vector<float2> &distProfileRestDistRanges,
    vector<int> &distProfileRestNumBins,
    vector<float4> &distProfileRestParams,
    vector<int2> &distProfileRestParamBounds,
    vector<float> &distProfileRestScaleFactor,
    vector<int> &distProfileRestGlobalIndices,
    vector<float> &restraintEnergies,
    vector<float3> &distProfileRestForces,
    int numDistProfileRestraints);

void computeTorsProfileRest(
    vector<RealVec> &pos,
    vector<int4> &torsProfileRestAtomIndices0,
    vector<int4> &torsProfileRestAtomIndices1,
    vector<int> &torsProfileRestNumBins,
    vector<float4> &torsProfileRestParams0,
    vector<float4> &torsProfileRestParams1,
    vector<float4> &torsProfileRestParams2,
    vector<float4> &torsProfileRestParams3,
    vector<int2> &torsProfileRestParamBounds,
    vector<float> &torsProfileRestScaleFactor,
    vector<int> &torsProfileRestGlobalIndices,
    vector<float> &restraintEnergies,
    vector<float3> &torsProfileRestForces,
    int numTorsProfileRestraints);

void computeGMMRest(
    vector<RealVec> &pos,
    int numGMMRestraints,
    vector<int4> &gmmParams,
    vector<int2> &gmmOffsets,
    vector<int> &gmmAtomIndices,
    vector<float> &gmmData,
    vector<float> &restraintEnergies,
    vector<float3> &gmmForces);

 void computeGridPotentialRest(
    vector<RealVec> &pos,
    vector<int> &atomIndices, 
    vector<float> &potentials,
    vector<float> &grid_x,
    vector<float> &grid_y,
    vector<float> &grid_z,
    vector<float> &weights,
    vector<int> &nxyz,
    vector<int> &densityIndices,
    vector<int> &indexToGlobal,
    int numRestraints,
    vector<float> &energies,    
    vector<float3> &forceBuffer);

void evaluateAndActivate(
    int numGroups,
    vector<int> &groupNumActive,
    vector<int2> &groupBounds,
    vector<int> &groupRestraintIndices,
    vector<float> &restraintEnergies,
    vector<bool> &restraintActive,
    vector<float> &groupEnergies);

void evaluateAndActivateCollections(
    int numCollections,
    vector<int> &collectionNumActive,
    vector<int2> &collectionBounds,
    vector<int> &collectionGroupIndices,
    vector<float> &groupEnergies,
    vector<bool> &groupActive);

void applyGroups(
    vector<bool> &groupActive,
    vector<bool> &restraintActive,
    vector<int2> &groupBounds,
    int numGroups);

float applyRDCRest(
    vector<RealVec>& force,
    vector<int2>& rdcRestAtomIndices,
    vector<int>& rdcAlignments,
    vector<int>& rdcRestGlobalIndices,
    vector<float3>& rdcRestForces,
    vector<float>& rdcRestDerivs,
    vector<float>& restraintEnergies,
    vector<bool>& restraintActive,
    std::map<std::string, double>& derivMap,
    int numRDCRestraints);

float applyDistRest(
    vector<RealVec> &force,
    vector<int2> &distanceRestAtomIndices,
    vector<int> &distanceRestGlobalIndices,
    vector<float3> &distanceRestForces,
    vector<float> &restraintEnergies,
    vector<bool> &restraintActive,
    int numDistRestraints);

float applyHyperbolicDistRest(
    vector<RealVec> &force,
    vector<int2> &hyperbolicDistanceRestAtomIndices,
    vector<int> &hyperbolicDistanceRestGlobalIndices,
    vector<float3> &hyperbolicDistanceRestForces,
    vector<float> &restraintEnergies,
    vector<bool> &restraintActive,
    int numHyperbolicDistRestraints);

float applyTorsionRest(
    vector<RealVec> &force,
    vector<int4> &torsionRestAtomIndices,
    vector<int> &torsionRestGlobalIndices,
    vector<float3> &torsionRestForces,
    vector<float> &restraintEnergies,
    vector<bool> &restraintActive,
    int numTorsionRestraints);

float applyDistProfileRest(
    vector<RealVec> &force,
    vector<int2> &distProfileRestAtomIndices,
    vector<int> &distProfileRestGlobalIndices,
    vector<float3> &distProfileRestForces,
    vector<float> &restraintEnergies,
    vector<bool> &restraintActive,
    int numDistProfileRestraints);

float applyTorsProfileRest(
    vector<RealVec> &force,
    vector<int4> &torsProfileRestAtomIndices0,
    vector<int4> &torsProfileRestAtomIndices1,
    vector<int> &torsProfileRestGlobalIndices,
    vector<float3> &torsProfileRestForces,
    vector<float> &restraintEnergies,
    vector<bool> &restraintActive,
    int numTorsProfileRestraints);

float applyGMMRest(
    vector<RealVec> &force,
    int numGMMRestraints,
    vector<int4> &gmmParams,
    vector<float> &restraintEnergies,
    vector<bool> &restraintActive,
    vector<int2> &gmmOffsets,
    vector<int> &gmmAtomIndices,
    vector<float3> &gmmForces);

float applyGridPotentialRest(
    vector<RealVec> &force,
    vector<int> &atomIndices,
    vector<int> &globalIndices,
    vector<float> &globalEnergies,
    vector<bool> &globalActive,
    vector<float3> &restForces,
    int numRestraints);
