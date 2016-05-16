/*
   Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
   All rights reserved
*/

#ifndef MELD_OPENMM_CUDAKERNELS_H_
#define MELD_OPENMM_CUDAKERNELS_H_

#include "meldKernels.h"
#include "openmm/kernels.h"
#include "openmm/System.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include "openmm/cuda/CudaSort.h"
#include <cufft.h>

namespace MeldPlugin {

class CudaCalcMeldForceKernel : public CalcMeldForceKernel {
public:
    CudaCalcMeldForceKernel(std::string name,
                                          const OpenMM::Platform& platform,
                                          OpenMM::CudaContext& cu,
                                          const OpenMM::System& system);
    ~CudaCalcMeldForceKernel();

    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the MeldForce this kernel will be used for
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

    void copyParametersToContext(OpenMM::ContextImpl& context, const MeldForce& force);

private:
    class ForceInfo;
    int numDistRestraints;
    int numHyperbolicDistRestraints;
    int numTorsionRestraints;
    int numDistProfileRestraints;
    int numDistProfileRestParams;
    int numTorsProfileRestraints;
    int numTorsProfileRestParams;
    int numRestraints;
    int numGroups;
    int numCollections;
    int largestGroup;
    int largestCollection;
    int groupsPerBlock;
    OpenMM::CudaContext& cu;
    const OpenMM::System& system;
    CUfunction computeDistRestKernel;
    CUfunction computeHyperbolicDistRestKernel;
    CUfunction computeTorsionRestKernel;
    CUfunction computeDistProfileRestKernel;
    CUfunction computeTorsProfileRestKernel;
    CUfunction evaluateAndActivateKernel;
    CUfunction evaluateAndActivateCollectionsKernel;
    CUfunction applyGroupsKernel;
    CUfunction applyDistRestKernel;
    CUfunction applyHyperbolicDistRestKernel;
    CUfunction applyTorsionRestKernel;
    CUfunction applyDistProfileRestKernel;
    CUfunction applyTorsProfileRestKernel;

    /**
     * Arrays for distance restraints
     *
     * Each array has size numDistRestraints
     */
    OpenMM::CudaArray* distanceRestRParams;       // float4 to hold r1-r4
    std::vector<float4> h_distanceRestRParams;

    OpenMM::CudaArray* distanceRestKParams;       // float to hold k
    std::vector<float> h_distanceRestKParams;

    OpenMM::CudaArray* distanceRestAtomIndices;   // int2 to hold i,j
    std::vector<int2> h_distanceRestAtomIndices;

    OpenMM::CudaArray* distanceRestGlobalIndices; // int to hold the global index for this restraint
    std::vector<int> h_distanceRestGlobalIndices;

    OpenMM::CudaArray* distanceRestForces; // cache to hold force computations until the final application step

    /**
     * Arrays for hyperbolic distance restraints
     *
     * Each array has size numHyperbolicDistRestraints
     */
    OpenMM::CudaArray* hyperbolicDistanceRestRParams;    // float4 to hold r1-r4
    std::vector<float4> h_hyperbolicDistanceRestRParams;

    OpenMM::CudaArray* hyperbolicDistanceRestParams;     // float4 to hold k1, k2, a, b
    std::vector<float4> h_hyperbolicDistanceRestParams;

    OpenMM::CudaArray* hyperbolicDistanceRestAtomIndices;   // int2 to hold i,j
    std::vector<int2> h_hyperbolicDistanceRestAtomIndices;

    OpenMM::CudaArray* hyperbolicDistanceRestGlobalIndices; // int to hold the global index for this restraint
    std::vector<int> h_hyperbolicDistanceRestGlobalIndices;

    OpenMM::CudaArray* hyperbolicDistanceRestForces; // cache to hold force computations until the final application step

    /**
     * Arrays for torsion restraints
     *
     * Each array has size numTorsionRestraints
     */
    OpenMM::CudaArray* torsionRestParams;           // float3 to hold phi, deltaPhi, forceConstant
    std::vector<float3> h_torsionRestParams;

    OpenMM::CudaArray* torsionRestAtomIndices;      // int4 to hold i,j,k,l
    std::vector<int4> h_torsionRestAtomIndices;

    OpenMM::CudaArray* torsionRestGlobalIndices;    // int to hold the global index for this restraint
    std::vector<int> h_torsionRestGlobalIndices;

    OpenMM::CudaArray* torsionRestForces;           // float3 * 4 to hold the forces on i,j,k,l for this restraint

    /**
     * Arrays for DistProfile restraints
     */
    OpenMM::CudaArray* distProfileRestAtomIndices; // int2 to hold i, j
    std::vector<int2> h_distProfileRestAtomIndices;

    OpenMM::CudaArray* distProfileRestDistRanges;  // float2 to hold rMin, rMax
    std::vector<float2> h_distProfileRestDistRanges;

    OpenMM::CudaArray* distProfileRestNumBins;     // int to hold the number of bins between rMin and rMax
    std::vector<int> h_distProfileRestNumBins;

    OpenMM::CudaArray* distProfileRestParamBounds; // int2 to hold the start and end of the parameter blocks for each rest
    std::vector<int2> h_distProileRestParamBounds;

    OpenMM::CudaArray* distProfileRestParams;      // float4 to hold a0..a3. There are NumBins of these for each rest
    std::vector<float4> h_distProfileRestParams;

    OpenMM::CudaArray* distProfileRestScaleFactor; // float to hold the scale factor for each restraint
    std::vector<float> h_distProfileRestScaleFactor;

    OpenMM::CudaArray* distProfileRestGlobalIndices;// int to hold the global index for each rest
    std::vector<int> h_distProfileRestGlobalIndices;

    OpenMM::CudaArray* distProfileRestForces;       // cache to hold the forces for each rest until the final application step

    /**
     * Arrays for TorsProfile restraints
     */
    OpenMM::CudaArray* torsProfileRestAtomIndices0; // int4to hold i, j, k, l for torsion 0
    std::vector<int4> h_torsProfileRestAtomIndices0;

    OpenMM::CudaArray* torsProfileRestAtomIndices1; // int4to hold i, j, k, l for torsion 1
    std::vector<int4> h_torsProfileRestAtomIndices1;

    OpenMM::CudaArray* torsProfileRestNumBins;     // int to hold the number of bins
    std::vector<int> h_torsProfileRestNumBins;

    OpenMM::CudaArray* torsProfileRestParamBounds; // int2 to hold the start and end of the parameter blocks for each rest
    std::vector<int2> h_torsProileRestParamBounds;

    OpenMM::CudaArray* torsProfileRestParams0;      // float4 to hold a0..a3. There are NumBins of these for each rest
    std::vector<float4> h_torsProfileRestParams0;
    OpenMM::CudaArray* torsProfileRestParams1;      // float4 to hold a4..a7. There are NumBins of these for each rest
    std::vector<float4> h_torsProfileRestParams1;
    OpenMM::CudaArray* torsProfileRestParams2;      // float4 to hold a8..a11. There are NumBins of these for each rest
    std::vector<float4> h_torsProfileRestParams2;
    OpenMM::CudaArray* torsProfileRestParams3;      // float4 to hold a12..a15. There are NumBins of these for each rest
    std::vector<float4> h_torsProfileRestParams3;

    OpenMM::CudaArray* torsProfileRestScaleFactor; // float to hold the scale factor for each restraint
    std::vector<float> h_torsProfileRestScaleFactor;

    OpenMM::CudaArray* torsProfileRestGlobalIndices;// int to hold the global index for each rest
    std::vector<int> h_torsProfileRestGlobalIndices;

    OpenMM::CudaArray* torsProfileRestForces;       // float3 * 8 to hold the forces on i, j, k, l, for this restraint

    /**
     * Arrays for all restraints
     *
     * Each array has size numRestraints
     */
    OpenMM::CudaArray* restraintEnergies;           // energy for each restraint

    OpenMM::CudaArray* restraintActive;             // is this restraint active?

    OpenMM::CudaArray* groupRestraintIndices;       // each group has bounds that index into this array, which gives restraints
    OpenMM::CudaArray* groupRestraintIndicesTemp;
    std::vector<int> h_groupRestraintIndices;

    /**
     * Arrays for all groups
     *
     * Each array has size numGroups
     */
    OpenMM::CudaArray* groupEnergies;                // energy for each group

    OpenMM::CudaArray* groupActive;                  // is this group active?

    OpenMM::CudaArray* groupBounds;                  // which range of groupRestraintIndices belongs to each group
    std::vector<int2> h_groupBounds;

    OpenMM::CudaArray* groupNumActive;               // number of restraints active for each group
    std::vector<int> h_groupNumActive;

    OpenMM::CudaArray* collectionGroupIndices;       // each collection has bounds that index into this array, giving groups
    std::vector<int> h_collectionGroupIndices;

    /**
     * Arrays for all collections
     *
     * Each array has size numCollections
     */
    OpenMM::CudaArray* collectionBounds;             // which range of collectionGroupIndices belongs to each collection
    std::vector<int2> h_collectionBounds;

    OpenMM::CudaArray* collectionNumActive;          // number of groups for each collection
    std::vector<int> h_collectionNumActive;

    OpenMM::CudaArray* collectionEnergies;

    OpenMM::CudaArray* collectionEncounteredNaN;     // flag that indicates that we encountered NaN when processing collections
    std::vector<int> h_collectionEncounteredNaN;

    void allocateMemory(const MeldForce& force);
    void setupDistanceRestraints(const MeldForce& force);
    void setupHyperbolicDistanceRestraints(const MeldForce& force);
    void setupTorsionRestraints(const MeldForce& force);
    void setupDistProfileRestraints(const MeldForce& force);
    void setupTorsProfileRestraints(const MeldForce& force);
    void setupGroups(const MeldForce& force);
    void setupCollections(const MeldForce& force);
    void validateAndUpload();
};


class CudaCalcRdcForceKernel : public CalcRdcForceKernel {
public:
    CudaCalcRdcForceKernel(std::string name,
                           const OpenMM::Platform& platform,
                           OpenMM::CudaContext& cu,
                           const OpenMM::System& system);

    ~CudaCalcRdcForceKernel();

    void initialize(const OpenMM::System& system, const RdcForce& force);

    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);

    void copyParametersToContext(OpenMM::ContextImpl& context, const RdcForce& force);

private:
    class ForceInfo;
    int numRdcRestraints;
    int numExperiments;
    OpenMM::CudaContext& cu;
    const OpenMM::System& system;
    CUfunction computeRdcPhase1;
    CUfunction computeRdcPhase3;

    /*
     * Per RDC data
     */
    OpenMM::CudaArray* r;                       // float4 array to hold direction cosines and norm of r
    OpenMM::CudaArray* atomExptIndices;         // int3 of indices of atoms involved in each dipole and exp't index
    OpenMM::CudaArray* lhs;                     // float 5 x numRdcRestraints array of the left-hand side of the LLS equations
    OpenMM::CudaArray* rhs;                     // float x numRdcRestraints array of observed couplings
    OpenMM::CudaArray* kappa;                   // float x numRedRestraints array of constants
    OpenMM::CudaArray* tolerance;               // float x numRdcRestraints array of tolerances
    OpenMM::CudaArray* force_const;             // float x numRdcRestraints array of force constants
    OpenMM::CudaArray* weight;                  // float x numRdcRestraints array of weights
    OpenMM::CudaArray* S;                       // float 5 x numExperiments array of alignment tensor elements
    // host vectors
    std::vector<int3> h_atomExptIndices;
    std::vector<int2> h_experimentBounds;
    std::vector<float> h_lhs;
    std::vector<float> h_rhs;
    std::vector<float> h_kappa;
    std::vector<float> h_tolerance;
    std::vector<float> h_force_const;
    std::vector<float> h_weight;
    std::vector<float> h_S;

    void allocateMemory(const RdcForce& force);
    void setupRdcRestraints(const RdcForce& force);
    void validateAndUpload();
    void computeRdcPhase2();
};

} // namespace MeldPlugin

#endif /*MELD_OPENMM_CUDAKERNELS_H*/
