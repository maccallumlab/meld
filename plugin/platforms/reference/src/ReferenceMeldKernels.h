#ifndef MELD_OPENMM_REFERENCEKERNELS_H_
#define MELD_OPENMM_REFERENCEKERNELS_H_

#include "MeldKernels.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/Platform.h"
#include "openmm/reference/RealVec.h"
#include <vector>
#include "MeldVecTypes.h"

namespace MeldPlugin {

class ReferenceCalcMeldForceKernel : public CalcMeldForceKernel {
public:
    ReferenceCalcMeldForceKernel(std::string name, const OpenMM::Platform& platform, const OpenMM::System& system);

    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the ExampleForce this kernel will be used for
     */
    void initialize(const OpenMM::System& system, const MeldForce& force);

    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);

    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the ExampleForce to copy the parameters from
     */
    void copyParametersToContext(OpenMM::ContextImpl& context, const MeldForce& force);

    void updateRDCGlobalParameters(OpenMM::ContextImpl& context);

private:
    int numRDCAlignments;
    int numRDCRestraints;
    int numDistRestraints;
    int numHyperbolicDistRestraints;
    int numTorsionRestraints;
    int numDistProfileRestraints;
    int numDistProfileRestParams;
    int numTorsProfileRestraints;
    int numTorsProfileRestParams;
    int numGMMRestraints;
    int numGridPotentials;
    int numGridPotentialRestraints;
    int numRestraints;
    int numGroups;
    int numCollections;
    float rdcScaleFactor;
    const OpenMM::System& system;
    /**
     * Arrays for RDC restraints.
     */
    std::vector<int> rdcRestAlignments;
    std::vector<float2> rdcRestParams1;
    std::vector<float3> rdcRestParams2;
    std::vector<int2> rdcRestAtomIndices;
    std::vector<int> rdcRestGlobalIndices;
    std::vector<float3> rdcRestForces;
    std::vector<float> rdcRestAlignmentComponents;
    std::vector<float> rdcRestDerivs;

    /**
     * Arrays for distance restraints
     *
     * Each array has size numDistRestraints
     */
    std::vector<float4> distanceRestRParams;
    std::vector<float> distanceRestKParams;
    std::vector<int2> distanceRestAtomIndices;
    std::vector<int> distanceRestGlobalIndices;
    std::vector<float3> distanceRestForces;

    /**
     * Arrays for hyperbolic distance restraints
     *
     * Each array has size numHyperbolicDistRestraints
     */
    std::vector<float4> hyperbolicDistanceRestRParams;
    std::vector<float4> hyperbolicDistanceRestParams;
    std::vector<int2> hyperbolicDistanceRestAtomIndices;
    std::vector<int> hyperbolicDistanceRestGlobalIndices;
    std::vector<float3> hyperbolicDistanceRestForces;

    /**
     * Arrays for torsion restraints
     *
     * Each array has size numTorsionRestraints
     */
    std::vector<float3> torsionRestParams;
    std::vector<int4> torsionRestAtomIndices;
    std::vector<int> torsionRestGlobalIndices;
    std::vector<float3> torsionRestForces;

    /**
     * Arrays for DistProfile restraints
     */
    std::vector<int2> distProfileRestAtomIndices;
    std::vector<float2> distProfileRestDistRanges;
    std::vector<int> distProfileRestNumBins;
    std::vector<int2> distProfileRestParamBounds;
    std::vector<float4> distProfileRestParams;
    std::vector<float> distProfileRestScaleFactor;
    std::vector<int> distProfileRestGlobalIndices;
    std::vector<float3> distProfileRestForces;

    /**
     * Arrays for TorsProfile restraints
     */
    std::vector<int4> torsProfileRestAtomIndices0;
    std::vector<int4> torsProfileRestAtomIndices1;
    std::vector<int> torsProfileRestNumBins;
    std::vector<int2> torsProfileRestParamBounds;
    std::vector<float4> torsProfileRestParams0;
    std::vector<float4> torsProfileRestParams1;
    std::vector<float4> torsProfileRestParams2;
    std::vector<float4> torsProfileRestParams3;
    std::vector<float> torsProfileRestScaleFactor;
    std::vector<int> torsProfileRestGlobalIndices;
    std::vector<float3> torsProfileRestForces;

    /**
     * Arrays for GMM restraints
     */
    std::vector<int4> gmmParams;
    std::vector<int2> gmmOffsets;
    std::vector<int> gmmAtomIndices;
    std::vector<float> gmmData;
    std::vector<float3> gmmForces;

    /**
     * Arrays for GridPotential
     * 
     */
    std::vector<float> gridPotentials;
    std::vector<float> gridPotentialgridx;
    std::vector<float> gridPotentialgridy;
    std::vector<float> gridPotentialgridz;
    std::vector<int> gridPotentialnxyz;   
    std::vector<int> gridPotentialRestAtomIndices;
    std::vector<int> gridPotentialRestGridPotentoalIndices;
    std::vector<float> gridPotentialRestWeights;
    std::vector<int> gridPotentialRestGlobalIndices;  
    std::vector<float3> gridPotentialRestForces;

    /**
     * Arrays for all restraints
     *
     * Each array has size numRestraints
     */
    std::vector<float> restraintEnergies;
    std::vector<bool> restraintActive;
    std::vector<int> groupRestraintIndices;

    /**
     * Arrays for all groups
     *
     * Each array has size numGroups
     */
    std::vector<float> groupEnergies;
    std::vector<bool> groupActive;
    std::vector<int2> groupBounds;
    std::vector<int> groupNumActive;
    std::vector<int> collectionGroupIndices;

    /**
     * Arrays for all collections
     *
     * Each array has size numCollections
     */
    std::vector<int2> collectionBounds;
    std::vector<int> collectionNumActive;
    std::vector<float> collectionEnergies;

    void setupRDCRestraints(const MeldForce& force);
    void setupDistanceRestraints(const MeldForce& force);
    void setupHyperbolicDistanceRestraints(const MeldForce& force);
    void setupTorsionRestraints(const MeldForce& force);
    void setupDistProfileRestraints(const MeldForce& force);
    void setupTorsProfileRestraints(const MeldForce& force);
    void setupGMMRestraints(const MeldForce& force);
    void setupGridPotentialRestraints(const MeldForce& force);
    void setupGroups(const MeldForce& force);
    void setupCollections(const MeldForce& force);
    int calcSizeGMMAtomIndices(const MeldForce& force);
    int calcSizeGMMData(const MeldForce& force);
    int3 calcNumGrids(const MeldForce& force);

    static std::map<std::string, double>& extractEnergyParameterDerivatives(OpenMM::ContextImpl& context);
};
}
#endif /*MELD_OPENMM_REFERENCEKERNELS_H*/
