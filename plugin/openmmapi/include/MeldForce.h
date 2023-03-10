/*
   Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
   All rights reserved
*/

#ifndef OPENMM_MELD_FORCE_H_
#define OPENMM_MELD_FORCE_H_

#include "openmm/Force.h"
#include "openmm/Vec3.h"
#include "internal/windowsExportMeld.h"
#include <map>
#include <vector>
#include <set>

namespace MeldPlugin {

/**
 * This is the MELD Force.
 */

class OPENMM_EXPORT_MELD MeldForce : public OpenMM::Force {

public:
    /**
     * Constructors
     */
     MeldForce();

     MeldForce(int numRDCAlignments, float rdcScaleFactor);

    /**
     * Update the per-restraint parameters in a Context to match those stored in this Force object.  This method provides
     * an efficient method to update certain parameters in an existing Context without needing to reinitialize it.
     * Simply call modifyDistanceRestaint(), modifyTorsionRestraint(), modifyDistProfileRestraint(),
     * or modifyTorsProfileRestraint() to modify the parameters of a restraint, then call updateParametersInContext()
     * to copy them over to the Context.
     *
     * This method has several limitations.  The only information it updates is the values of per-restraint parameters.
     * All other aspects of the Force (such as the energy function) are unaffected and can only be changed by reinitializing
     * the Context.  The set of particles involved in a restraint cannot be changed, nor can new restraints be added.
     */
    void updateParametersInContext(OpenMM::Context& context);

    /**
     * @return The number of RDC alignment tensors in this force.
     */ 
    int getNumRDCAlignments() const;

    /**
     * @return The RDC scale factor.
     */
    float getRDCScaleFactor() const;

    /**
     * @return A bool indicating if particle is involved in MELD force
     */
    bool containsParticle(int particle) const;

    /**
     * @return The number of RDC restraints.
     */
    int getNumRDCRestraints() const;

    /**
     * @return The number of distance restraints.
     */
    int getNumDistRestraints() const;

    /**
     * @return The number of hyperbolic distance restraints.
     */
    int getNumHyperbolicDistRestraints() const;

    /**
     * @return The number of torsion restraints.
     */
    int getNumTorsionRestraints() const;

    /**
     * @return The number of distance profile restraints.
     */
    int getNumDistProfileRestraints() const;

    /**
     * @return The number of distance profile restraint parameters.
     */
    int getNumDistProfileRestParams() const;

    /**
     * @return The number of torsion profile restraints.
     */
    int getNumTorsProfileRestraints() const;

    /**
     * @return The number of torsion profile restraint parameters.
     */
    int getNumTorsProfileRestParams() const;

    /**
     * @return The number of GMM restraints
     */
    int getNumGMMRestraints() const;
 
    /**
     * @return The number of grid potentials
     */
    int getNumGridPotentials() const;

    /**
     * @return The number of grid potential restraints
     */
    int getNumGridPotentialRestraints() const;

    /**
     * @return The total number of distance and torsion restraints.
     */
    int getNumTotalRestraints() const;

    /**
     * @return The number of restraint groups.
     */
    int getNumGroups() const;

    /**
     * @return The number of collections of restraint groups.
     */
    int getNumCollections() const;

    /**
     * Get the parameters of an RDC restraint.
     *
     * @param index the index to retrieve
     * @param particle1 the first particle in the RDC
     * @param particle2 the second particle in the RDC
     * @param alignment the index of the alignment tensor to use
     * @param kappa the kappa parameter in Hz nm^3
     * @param obs the observed dipolar coupling in Hz
     * @param tol the tolerance in Hz
     * @param quad_cut number of Hz for transition from quadratic to linear
     * @param force_constant the force constant in kJ mol^-1 Hz^-2
     * @param globalIndex the global index of the restraint
     */
    void getRDCRestraintParameters(int index, int& particle1, int& particle2, int& alignment,
                                   float& kappa, float& obs, float& tol, float& quad_cut, float& force_constant,
                                   int& globalIndex) const;

    /**
     * Get the parameters for a distance restraint. See addDistanceRestraint()
     * for more details about the parameters.
     *
     * @param index  the index of the restraint
     * @param atom1  the first atom
     * @param atom2  the second atom
     * @param r1  the upper bound of region 1
     * @param r2  the upper bound of region 2
     * @param r3  the upper bound of region 3
     * @param r4  the upper bound of region 4
     * @param forceConstant  the force constant
     * @param globalIndex  the global index of the restraint
     */
    void getDistanceRestraintParams(int index, int& atom1, int& atom2, float& r1, float& r2, float& r3,
            float& r4, float& forceConstant, int& globalIndex) const;

    /**
     * Get the parameters for a hyperbolic distance restraint. See addHyperbolicDistanceRestraint()
     * for more details about the parameters.
     *
     * @param index  the index of the restraint
     * @param atom1  the first atom
     * @param atom2  the second atom
     * @param r1  the upper bound of region 1
     * @param r2  the upper bound of region 2
     * @param r3  the upper bound of region 3
     * @param r4  the upper bound of region 4
     * @param forceConstant  the force constant for region 1
     * @param asymptote the asymptotic energy in region 4
     * @param globalIndex  the global index of the restraint
     */

    void getHyperbolicDistanceRestraintParams(int index, int& atom1, int& atom2, float& r1, float& r2, float& r3,
            float& r4, float& forceConstant, float& asymptote, int& globalIndex) const;
    /**
     * Get the parameters for a torsion restraint. See addTorsionRestraint() for
     * more details about the parameters.
     *
     * @param index  the index of the restraint
     * @param atom1  the first atom
     * @param atom2  the second atom
     * @param atom3  the third atom
     * @param atom4  the fourth atom
     * @param phi  the equilibrium torsion (degrees)
     * @param deltaPhi  the deltaPhi parameter (degrees)
     * @param forceConstant  the force constant
     * @param globalIndex  the global index of the restraint
     */
    void getTorsionRestraintParams(int index, int& atom1, int& atom2, int& atom3, int&atom4,
            float& phi, float& deltaPhi, float& forceConstant, int& globalIndex) const;

    /**
     * Get the parameters for a distance profile restraint. See addDistProfileRestraint()
     * for more details about the parameters.
     *
     * @param index  the index of the restraint
     * @param atom1  the first atom
     * @param atom2  the second atom
     * @param rMin  the lower bound of the restraint
     * @param rMax  the upper bound of the restraint
     * @param nBins  the number of bins
     * @param aN  the Nth spline parameter where N is in (0,1,2,3)
     * @param scaleFactor  the scale factor
     * @param globalIndex  the global index of the restraint
     */
    void getDistProfileRestraintParams(int index, int& atom1, int& atom2, float& rMin, float & rMax,
            int& nBins, std::vector<double>& a0, std::vector<double>& a1, std::vector<double>& a2,
            std::vector<double>& a3, float& scaleFactor, int& globalIndex) const;

    /**
     * Get the parameters for a torsion profile restraint.
     *
     * @param index  the index of the restraint
     * @param atom1  the first atom
     * @param atom2  the second atom
     * @param atom3  the third atom
     * @param atom4  the fourth atom
     * @param atom5  the fifth atom
     * @param atom6  the sixth atom
     * @param atom7  the seventh atom
     * @param atom8  the eighth atom
     * @param nBins  the number of bins
     * @param aN  the Nth spline parameter where N is in (0,1,...,14,15)
     * @param scaleFactor  the scale factor
     * @param globalIndex  the global index of the restraint
     */
    void getTorsProfileRestraintParams(int index, int& atom1, int& atom2, int& atom3, int& atom4,
            int& atom5, int& atom6, int& atom7, int& atom8, int& nBins,
            std::vector<double>&  a0, std::vector<double>&  a1, std::vector<double>&  a2,
            std::vector<double>&  a3, std::vector<double>&  a4, std::vector<double>&  a5,
            std::vector<double>&  a6, std::vector<double>&  a7, std::vector<double>&  a8,
            std::vector<double>&  a9, std::vector<double>& a10, std::vector<double>& a11,
            std::vector<double>& a12, std::vector<double>& a13, std::vector<double>& a14,
            std::vector<double>& a15, float& scaleFactor, int& globalIndex) const;

    /**
     * Get the parameters for a GMM restraint. See addGMMRestraint()
     * for more details about the parameters.
     *
     * @param index        the index of the restraint
     * @param nPairs       the number of atom pairs
     * @param nComponents  the number of GMM components
     * @param scale        the overall scaling applied to the forces and energies
     * @param atomIndices  the vector of atom indices
     * @param weights      the vector of weights
     * @param means        the vector of means in nm
     * @param precisionOnDiagonal    the diagonals of the precision matrix in nm^(-2)
     * @param precisionOffDiagonal   the off-diagonals of the precision matrix in nm^(-2)
     * @param globalIndex  the global index of the restraint
     */
    void getGMMRestraintParams(int index, int& nPairs, int& nComponents, float& scale,
                               std::vector<int>& atomIndices,
                               std::vector<double>& weights,
                               std::vector<double>& means,
                               std::vector<double>& precisionOnDiagonal,
                               std::vector<double>& precisionOffDiagonal,
                               int& globalIndex) const;

    /**
     * Get the parameters for a group of restraints.
     *
     * @param index  the index of the group
     * @param indices  the indices of the restraints in the group
     * @param numActive  the number of active restraints in the group
     */
    void getGroupParams(int index, std::vector<int>& indices, int& numActive) const;

    /**
     * Get the parameters for a collection of restraint groups.
     *
     * @param index  the index of the collection
     * @param indices  the indices of the groups in the collection
     * @param numActive  the number of active groups in the collection
     */
    void getCollectionParams(int index, std::vector<int>& indices, int& numActive) const;

    void getGridPotentialParams(int index, std::vector<double>& potential,float& originx,float& originy,float& originz,
            float& gridx,float& gridy,float& gridz, int& nx, int& ny, int& nz) const;
    

    void getGridPotentialRestraintParams(int index, std::vector<int>& atom, 
                            std::vector<double>& mu, 
                            std::vector<double>& gridpos_x,
                            std::vector<double>& gridpos_y,
                            std::vector<double>& gridpos_z,
                            int& globalIndex) const;
    /**
     * @param particle1 the first particle in the RDC
     * @param particle2 the second particle in the RDC
     * @param alignment the index of the alignment tensor to use
     * @param kappa the kappa parameter in Hz nm^3
     * @param obs the observed dipolar coupling in Hz
     * @param tol the tolerance in Hz
     * @param quad_cut number of Hz for transition from quadratic to linear
     * @param force_constant the force constant in kJ mol^-1 Hz^-2
     * @return the index of the restraint that was created
     */
    int addRDCRestraint(int particle1, int particle2, int alignment,
                        float kappa, float obs, float tol, float quad_cut, float force_constant);

    /**
     * @param index the index of the restraint to modify
     * @param particle1 the first particle in the RDC
     * @param particle2 the second particle in the RDC
     * @param alignment the index of the alignment tensor to use
     * @param kappa the kappa parameter in Hz nm^3
     * @param obs the observed dipolar coupling in Hz
     * @param tol the tolerance in Hz
     * @param quad_cut number of Hz for transition from quadratic to linear
     * @param force_constant the force constant in kJ mol^-1 Hz^-2
     */
    void modifyRDCRestraint(int index, int particle1, int particle2, int alignment,
                            float kappa, float obs, float tol, float quad_cut, float force_constant);
    /**
     * Create a new distance restraint.
     * There are five regions:
     *
     * I:    r < r1
     *
     * II:  r1 < r < r2
     *
     * III: r2 < r < r3
     *
     * IV:  r3 < r < r4
     *
     * V:   r4 < r
     *
     * The energy is linear in regions I and V, quadratic in II and IV, and zero in III.
     * 
     * There is special handling of the restraints if particle1=-1. In this case,
     * the energy and force will be zero. However, when selecting active restraints
     * within a group, the energy will be treated as MAX_FLOAT. This behavior is
     * to facilitate support for peak mapping, which needs a way for a restraint
     * to be effectively ignored.
     *
     * @param particle1  the first atom
     * @param particle2  the second atom
     * @param r1  the upper bound of region 1
     * @param r2  the upper bound of region 2
     * @param r3  the upper bound of region 3
     * @param r4  the upper bound of region 4
     * @param forceConstant  the force constant
     * @return the index of the restraint that was created
     */
    int addDistanceRestraint(int particle1, int particle2, float r1, float r2, float r3, float r4,
            float force_constant);

    /**
     * Modify an existing distance restraint. See addDistanceRestraint() for more
     * details about the parameters.
     *
     * @param index  the index of the restraint
     * @param particle1  the first atom
     * @param particle2  the second atom
     * @param r1  the upper bound of region 1
     * @param r2  the upper bound of region 2
     * @param r3  the upper bound of region 3
     * @param r4  the upper bound of region 4
     * @param forceConstant  the force constant
     */
    void modifyDistanceRestraint(int index, int particle1, int particle2, float r1, float r2, float r3,
            float r4, float force_constant);

    /**
     * Create a new hyperbolic distance restraint.
     * There are five regions:
     *
     * I:    r < r1
     *
     * II:  r1 < r < r2
     *
     * III: r2 < r < r3
     *
     * IV:  r3 < r < r4
     *
     * V:   r4 < r
     *
     * The energy is linear in region I, quadratic in II and IV, and zero in III.
     *
     * The energy is hyperbolic in region V, with an asymptotic value set by the
     * parameter asymptote. The energy will be 1/3 of the asymptotic value at r=r4.
     * The distance between r3 and r4 controls the steepness of the potential.
     *
     * @param particle1  the first atom
     * @param particle2  the second atom
     * @param r1  the upper bound of region 1
     * @param r2  the upper bound of region 2
     * @param r3  the upper bound of region 3
     * @param r4  the upper bound of region 4
     * @param forceConstant  the force constant in regions I and II
     * @param asymptote the asymptotic value in region V, also controls the steepness in region IV.
     * @return the index of the restraint that was created
     */
    int addHyperbolicDistanceRestraint(int particle1, int particle2, float r1, float r2, float r3, float r4,
            float force_constant, float asymptote);

    /**
     * Modify an existing hyperbolic distance restraint. See addHyperbolicDistanceRestraint() for more
     * details about the parameters.
     *
     * @param index  the index of the restraint
     * @param particle1  the first atom
     * @param particle2  the second atom
     * @param r1  the upper bound of region 1
     * @param r2  the upper bound of region 2
     * @param r3  the upper bound of region 3
     * @param r4  the upper bound of region 4
     * @param forceConstant  the force constant
     * @param asymptote the asymptotic value
     */
    void modifyHyperbolicDistanceRestraint(int index, int particle1, int particle2, float r1, float r2, float r3,
            float r4, float force_constant, float asymptote);

    /**
     * Create a new torsion restraint.
     *
     * If (x - phi) < -deltaPhi:
     *    E = 1/2 * forceConstant * (x - phi + deltaPhi)^2
     *
     * Else if (x - phi) > deltaPhi:
     *    E = 1/2 * forceConstant * (x - phi - deltaPhi)^2
     *
     * Else:
     *    E = 0
     *
     * @param atom1  the first atom
     * @param atom2  the second atom
     * @param atom3  the third atom
     * @param atom4  the fourth atom
     * @param phi  the equilibrium torsion (degrees)
     * @param deltaPhi  the deltaPhi parameter (degrees)
     * @param forceConstant  the force constant
     * @return the index of the restraint that was created
     */
    int addTorsionRestraint(int atom1, int atom2, int atom3, int atom4, float phi, float deltaPhi, float forceConstant);

    /**
     * Modify an existing torsion restraint. See addTorsionRestraint() for more
     * details about the parameters.
     *
     * @param index  the index of the restraint
     * @param atom1  the first atom
     * @param atom2  the second atom
     * @param atom3  the third atom
     * @param atom4  the fourth atom
     * @param phi  the equilibrium torsion (degrees)
     * @param deltaPhi  the deltaPhi parameter (degrees)
     * @param forceConstant  the force constant
     */
    void modifyTorsionRestraint(int index, int atom1, int atom2, int atom3, int atom4, float phi,
            float deltaPhi, float forceConstant);

    /**
     * Create a new distance profile restraint.
     *
     * bin = floor( (r - rMin) / (rMax - rMin) * nBins) )
     *
     * binWidth = (rMax - rMin) / nBins
     *
     * t = (r - bin * binWidth + rMin) / binWidth;
     *
     * E = scaleFactor * (a0 + a1 * t + a2 * t^2 + a3 * t^3)
     *
     * @param atom1  the first atom
     * @param atom2  the second atom
     * @param rMin  the lower bound of the restraint
     * @param rMax  the upper bound of the restraint
     * @param nBins  the number of bins
     * @param aN  the Nth spline parameter where N is in (0,1,2,3)
     * @param scaleFactor  the scale factor
     * @return the index of the restraint that was created
     */
    int addDistProfileRestraint(int atom1, int atom2, float rMin, float rMax, int nBins, std::vector<double> a0,
            std::vector<double> a1, std::vector<double> a2, std::vector<double> a3, float scaleFactor);

    /**
     * Modify an existing distance profile restraint. See addDistProfileRestraint()
     * for more details about the parameters.
     *
     * @param index  the index of the restraint
     * @param atom1  the first atom
     * @param atom2  the second atom
     * @param rMin  the lower bound of the restraint
     * @param rMax  the upper bound of the restraint
     * @param nBins  the number of bins
     * @param aN  the Nth spline parameter where N is in (0,1,2,3)
     * @param scaleFactor  the scale factor
     */
    void modifyDistProfileRestraint(int index, int atom1, int atom2, float rMin, float rMax, int nBins,
            std::vector<double> a0, std::vector<double> a1, std::vector<double> a2, std::vector<double> a3,
            float scaleFactor);

    /**
     * Create a new torsion profile restraint.
     *
     * @param atom1  the first atom
     * @param atom2  the second atom
     * @param atom3  the third atom
     * @param atom4  the fourth atom
     * @param atom5  the fifth atom
     * @param atom6  the sixth atom
     * @param atom7  the seventh atom
     * @param atom8  the eighth atom
     * @param nBins  the number of bins
     * @param aN  the Nth spline parameter where N is in (0,1,...,14,15)
     * @param scaleFactor  the scale factor
     * @return the index of the restraint that was created
     */
    int addTorsProfileRestraint(int atom1, int atom2, int atom3, int atom4,
            int atom5, int atom6, int atom7, int atom8, int nBins,
            std::vector<double>  a0, std::vector<double>  a1, std::vector<double>  a2,
            std::vector<double>  a3, std::vector<double>  a4, std::vector<double>  a5,
            std::vector<double>  a6, std::vector<double>  a7, std::vector<double>  a8,
            std::vector<double>  a9, std::vector<double> a10, std::vector<double> a11,
            std::vector<double> a12, std::vector<double> a13, std::vector<double> a14,
            std::vector<double> a15, float scaleFactor);

    /**
     * Modify an existing torsion profile restraint.
     *
     * @param index  the index of the restraint
     * @param atom1  the first atom
     * @param atom2  the second atom
     * @param atom3  the third atom
     * @param atom4  the fourth atom
     * @param atom5  the fifth atom
     * @param atom6  the sixth atom
     * @param atom7  the seventh atom
     * @param atom8  the eighth atom
     * @param nBins  the number of bins
     * @param aN  the Nth spline parameter where N is in (0,1,...,14,15)
     * @param scaleFactor  the scale factor
     */
    void modifyTorsProfileRestraint(int index, int atom1, int atom2, int atom3, int atom4,
            int atom5, int atom6, int atom7, int atom8, int nBins,
            std::vector<double>  a0, std::vector<double>  a1, std::vector<double>  a2,
            std::vector<double>  a3, std::vector<double>  a4, std::vector<double>  a5,
            std::vector<double>  a6, std::vector<double>  a7, std::vector<double>  a8,
            std::vector<double>  a9, std::vector<double> a10, std::vector<double> a11,
            std::vector<double> a12, std::vector<double> a13, std::vector<double> a14,
            std::vector<double> a15, float scaleFactor);

    /**
     * Create a new GMM restraint.
     *
     * This is a Gaussian Mixture Model restraint involving a series of
     * distances.
     *
     * The energy has the form:
     *
     * E = w1 N1 exp(-0.5 (r-u1)^T P1 (r-u1)) + w2 N2 exp(-0.5 (r-u2)^T P2 (r-u2)) + ...
     *
     * where:
     *    w1, w2, ... are the nComponents weights,
     *    N1, N2, ... are automatically calculated normalization factors
     *    r is the vector of distances for the atom pairs in atomIndices
     *    u1, u2, ... are the mean vectors for each component
     *    P1, P2, ... are the precision (inverse covariance) matrices for each component
     *
     * The memory layout is as follows:
     * atomIndices -> [pair1_atom_1, pair1_atom_2, pair2_atom_1, pair2_atom_2, ...]
     * weights -> [wa, wb, ...]
     * means -> [ma1, ma2, ..., mb1, mb2, ...]
     * precisionOnDiagonal -> [Pa11, Pa22, ..., Pb11, Pb22, ...]
     * precisionOffDiagonal -> [Pa12, Pa13, ...,  Pa23, ..., Pb12, Pb13, ..., Pb23, ...]
     *
     * where a, b, ... are the different components and 1, 2, ... are the different distances.
     *
     * atomIndices.size() must be 2 * nPairs
     * weights.size() must be nComponents
     * means.size() must be nPairs * nComponents
     * precisionOnDiagonal.size() must be nPairs * nComponents
     * precisionOffDiagonal.size() must be nPairs * nComponents * (nComponents - 1) / 2
     *
     * @param nPairs       the number of atom pairs
     * @param nComponents  the number of GMM components
     * @param scale        the overall scaling applied to the forces and energies
     * @param atomIndices  the vector of atom indices
     * @param weights      the vector of weights
     * @param means        the vector of means in nm
     * @param precisionOnDiagonal    the diagonals of the precision matrix in nm^(-2)
     * @param precisionOffDiagonal   the off-diagonals of the precision matrix in nm^(-2)
     * @return the index of the restraint that was created
     */
    int addGMMRestraint(int nPairs, int nComponents, float scale,
                        std::vector<int> atomIndices,
                        std::vector<double> weights,
                        std::vector<double> means,
                        std::vector<double> precisionOnDiagonal,
                        std::vector<double> precisionOffDiagonal);

    /**
     * Modify an existing GMM restraint. See addGMMRestraint() for more
     * details about the parameters.
     *
     * @param index  the index of the restraint
     * @param nPairs       the number of atom pairs
     * @param nComponents  the number of GMM components
     * @param scale        the overall scaling applied to the forces and energies
     * @param atomIndices  the vector of atom indices
     * @param weights      the vector of weights
     * @param means        the vector of means in nm
     * @param precisionOnDiagonal    the diagonals of the precision matrix in nm^(-2)
     * @param precisionOffDiagonal   the off-diagonals of the precision matrix in nm^(-2)
     */
    void modifyGMMRestraint(int index, int nPairs, int nComponents, float scale,
                            std::vector<int> atomIndices,
                            std::vector<double> weights,
                            std::vector<double> means,
                            std::vector<double> precisionOnDiagonal,
                            std::vector<double> precisionOffDiagonal);

    /**
     * Create a new group of restraints.
     *
     * @param restraint_indices  the indices of the restraints in the group
     * @param n_active  the number of active restraints in the group
     * @return the index of the group that was created
     */
    int addGroup(std::vector<int> restraint_indices, int n_active);

    /**
     * Modify the number of active restraints in a group.
     * 
     * @param index the index of the group to modify
     * @param n_active the new number of active restraints
     */
    void modifyGroupNumActive(int index, int n_active);

    /**
     * Create a new collection of restraint groups.
     *
     * @param group_indices  the indices of the groups in the collection
     * @param n_active  the number of active groups in the collection
     * @return the index of the collection that was created
     */
    int addCollection(std::vector<int> group_indices, int n_active);

    /**
     * Modify the number of active groups in a collection.
     * 
     * @param index the index of the collection to modify
     * @param n_active the new number of active groups
     */
    void modifyCollectionNumActive(int index, int n_active);

    bool usesPeriodicBoundaryConditions() const override;

    /**
     * Add a new grid potential to the system
     */
    void addGridPotential(
        std::vector<double> potential,
        float originx,
        float originy,
        float originz,
        float gridx,
        float gridy,
        float gridz,
        int nx,
        int ny,
        int nz,
        int density_index);

    /**
     * Modify a grid potential
     */
    void modifyGridPotential(
        int index, 
        std::vector<double> potential,
        float originx,
        float originy,
        float originz,
        float gridx,
        float gridy,
        float gridz,
        int nx,
        int ny,
        int nz);

    /**
     * Add a grid potential restraint
     */
    int addGridPotentialRestraint(std::vector<int> particle, std::vector<double> mu,
        std::vector<double> gridpos_x, std::vector<double> gridpos_y, std::vector<double> gridpos_z);

    void modifyGridPotentialRestraint(int index, std::vector<int> particle, std::vector<double> mu,
        std::vector<double> gridpos_x, std::vector<double> gridpos_y, std::vector<double> gridpos_z);


    std::vector<std::pair<int, int> > getBondedParticles() const;

protected:
    OpenMM::ForceImpl* createImpl() const;

private:
    class RDCRestraintInfo;
    class DistanceRestraintInfo;
    class TorsionRestraintInfo;
    class HyperbolicDistanceRestraintInfo;
    class DistProfileRestraintInfo;
    class TorsProfileRestraintInfo;
    class GMMRestraintInfo;
    class GridPotentialInfo;
    class GridPotentialRestraintInfo;
    class GroupInfo;
    class CollectionInfo;
    int n_restraints;
    int n_rdc_alignments;
    float rdcScaleFactor;
    std::vector<RDCRestraintInfo> rdcRestraints;
    std::vector<DistanceRestraintInfo> distanceRestraints;
    std::vector<HyperbolicDistanceRestraintInfo> hyperbolicDistanceRestraints;
    std::vector<TorsionRestraintInfo> torsions;
    std::vector<DistProfileRestraintInfo> distProfileRestraints;
    std::vector<TorsProfileRestraintInfo> torsProfileRestraints;
    std::vector<GMMRestraintInfo> gmmRestraints;
    std::vector<GridPotentialInfo> gridPotentials;
    std::vector<GridPotentialRestraintInfo> gridPotentialRestraints;
    std::vector<GroupInfo> groups;
    std::vector<CollectionInfo> collections;
    std::set<int> meldParticleSet;
    bool isDirty;
    void updateMeldParticleSet();

    class RDCRestraintInfo {
    public:
        int particle1, particle2;
        int alignment;
        float kappa, obs;
        float tol, quad_cut, force_constant;
        int global_index;

        RDCRestraintInfo() {
            particle1 = particle2 = alignment = -1;
            kappa = obs = tol = quad_cut = force_constant = 0.0;
            global_index = -1;
        }

        RDCRestraintInfo(int particle1, int particle2, int alignment,
                         float kappa, float obs, float tol, float quad_cut, float force_constant,
                         int global_index):
            particle1(particle1), particle2(particle2), alignment(alignment),
            kappa(kappa), obs(obs), tol(tol), quad_cut(quad_cut), force_constant(force_constant),
            global_index(global_index) {
        }
    };

    class DistanceRestraintInfo {
    public:
        int particle1, particle2;
        float r1, r2, r3, r4, force_constant;
        int global_index;

        DistanceRestraintInfo() {
            particle1 = particle2    = -1;
            force_constant = 0.0;
            r1 = r2 = r3 = r4 = 0.0;
            global_index = -1;
        }

        DistanceRestraintInfo(int particle1, int particle2, float r1, float r2, float r3, float r4,
                float force_constant, int global_index) : particle1(particle1), particle2(particle2), r1(r1),
                                                            r2(r2), r3(r3), r4(r4), force_constant(force_constant),
                                                            global_index(global_index) {
        }
    };

    class HyperbolicDistanceRestraintInfo {
    public:
        int particle1, particle2;
        float r1, r2, r3, r4, force_constant, asymptote;
        int global_index;

        HyperbolicDistanceRestraintInfo() {
            particle1 = particle2    = -1;
            force_constant = 0.0;
            r1 = r2 = r3 = r4 = asymptote = 0.0;
            global_index = -1;
        }

        HyperbolicDistanceRestraintInfo(int particle1, int particle2, float r1, float r2, float r3, float r4,
                float force_constant, float asymptote, int global_index) : particle1(particle1), particle2(particle2), r1(r1),
                                                            r2(r2), r3(r3), r4(r4), force_constant(force_constant),
                                                            asymptote(asymptote), global_index(global_index) {
        }
    };

    class TorsionRestraintInfo {
    public:
        int atom1, atom2, atom3, atom4;
        float phi, deltaPhi, forceConstant;
        int globalIndex;

        TorsionRestraintInfo() {
            atom1 = atom2 = atom3 = atom4 = -1;
            phi = 0;
            deltaPhi = 0;
            forceConstant = 0;
            globalIndex =  -1;
        }

        TorsionRestraintInfo(int atom1, int atom2, int atom3, int atom4, float phi, float deltaPhi,
                               float forceConstant, int globalIndex) :
            atom1(atom1), atom2(atom2), atom3(atom3), atom4(atom4), phi(phi), deltaPhi(deltaPhi),
            forceConstant(forceConstant), globalIndex(globalIndex) {
        }
    };

    class DistProfileRestraintInfo {
    public:
        int atom1, atom2, nBins;
        float rMin, rMax, scaleFactor;
        std::vector<double> a0;
        std::vector<double> a1;
        std::vector<double> a2;
        std::vector<double> a3;
        int globalIndex;

        DistProfileRestraintInfo() {
            atom1 = atom2 = -1;
            nBins = 0;
            rMin = rMax = scaleFactor = -1.0;
            globalIndex = -1;
        }

        DistProfileRestraintInfo(int atom1, int atom2, float rMin, float rMax, int nBins,
                std::vector<double> a0, std::vector<double> a1, std::vector<double> a2,
                std::vector<double> a3, float scaleFactor, int globalIndex) :
            atom1(atom1), atom2(atom2), nBins(nBins), rMin(rMin), rMax(rMax), scaleFactor(scaleFactor),
            globalIndex(globalIndex), a0(a0), a1(a1), a2(a2), a3(a3) {
            }
    };

    class TorsProfileRestraintInfo {
    public:
        int atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8, nBins, globalIndex;
        float scaleFactor;
        std::vector<double> a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15;

        TorsProfileRestraintInfo() {
            atom1 = atom2 = atom3 = atom4 = atom5 = atom6 = atom7 = atom8 = nBins = globalIndex = -1;
            scaleFactor = 0;
        }

        TorsProfileRestraintInfo(int atom1, int atom2, int atom3, int atom4,
                int atom5, int atom6, int atom7, int atom8, int nBins,
                std::vector<double>  a0, std::vector<double>  a1, std::vector<double>  a2,
                std::vector<double>  a3, std::vector<double>  a4, std::vector<double>  a5,
                std::vector<double>  a6, std::vector<double>  a7, std::vector<double>  a8,
                std::vector<double>  a9, std::vector<double> a10, std::vector<double> a11,
                std::vector<double> a12, std::vector<double> a13, std::vector<double> a14,
                std::vector<double> a15, float scaleFactor, int globalIndex) :
            atom1(atom1), atom2(atom2), atom3(atom3), atom4(atom4),
            atom5(atom5), atom6(atom6), atom7(atom7), atom8(atom8), nBins(nBins),
            a0(a0), a1(a1), a2(a2), a3(a3), a4(a4), a5(a5), a6(a6), a7(a7),
            a8(a8), a9(a9), a10(a10), a11(a11), a12(a12), a13(a13), a14(a14), a15(a15),
            scaleFactor(scaleFactor), globalIndex(globalIndex) {
            }
    };

    class GroupInfo {
    public:
        std::vector<int> restraint_indices;
        int n_active;

        GroupInfo(): n_active(0) {
        }

        GroupInfo(std::vector<int> restraint_indices, int n_active):
            restraint_indices(restraint_indices), n_active(n_active) {
        }
    };

    class CollectionInfo {
    public:
        std::vector<int> group_indices;
        int n_active;

        CollectionInfo(): n_active(0) {
        }

        CollectionInfo(std::vector<int> group_indices, int n_active) :
            group_indices(group_indices), n_active(n_active) {
        }
    };

    class GMMRestraintInfo {
    public:
      int nPairs, nComponents, globalIndex;
      float scale;
      std::vector<int> atomIndices;
      std::vector<double> weights;
      std::vector<double> means;
      std::vector<double> precisionOnDiagonal;
      std::vector<double> precisionOffDiagonal;

      GMMRestraintInfo() {
        nPairs = 0;
        nComponents = 0;
        globalIndex = -1;
        scale = 1.0;
      }

    GMMRestraintInfo(int nPairs, int nComponents, int globalIndex, float scale,
                     std::vector<int> atomIndices,
                     std::vector<double> weights, std::vector<double> means,
                     std::vector<double> precisionOnDiagonal,
                     std::vector<double> precisionOffDiagonal) :
      nPairs(nPairs), nComponents(nComponents), globalIndex(globalIndex), scale(scale),
      weights(weights), means(means), precisionOffDiagonal(precisionOffDiagonal),
      precisionOnDiagonal(precisionOnDiagonal), atomIndices(atomIndices) {}
    };

    class GridPotentialInfo {
    public:
        std::vector<double> potential;
        double originx;
        double originy;
        double originz;
        double gridx;
        double gridy;
        double gridz;
        int nx;
        int ny;
        int nz;
        int density_index;

        GridPotentialInfo() {
            originx = 0.0;
            originy = 0.0;
            originz = 0.0;
            gridx = 0.0;
            gridy = 0.0;
            gridz = 0.0;
            nx = -1;
            ny = -1;
            nz = -1;
            density_index = -1;
        }

        GridPotentialInfo(std::vector<double> potential,double originx,double originy,double originz,
        double gridx,double gridy,double gridz,int nx,int ny,int nz, int density_index):
        potential(potential), originx(originx), originy(originy), originz(originz), 
        gridx(gridx), gridy(gridy), gridz(gridz), nx(nx), ny(ny), nz(nz), density_index(density_index) {}
    };

    class GridPotentialRestraintInfo {
    public:
        std::vector<int> particle;
        std::vector<double> mu;
        std::vector<double> gridpos_x;
        std::vector<double> gridpos_y; 
        std::vector<double> gridpos_z;
        int globalIndex;

        GridPotentialRestraintInfo() {
            globalIndex = -1;
        }
        GridPotentialRestraintInfo(std::vector<int> particle,  std::vector<double> mu, 
        std::vector<double> gridpos_x, std::vector<double> gridpos_y, std::vector<double> gridpos_z, int globalIndex): 
        particle(particle), mu(mu), gridpos_x(gridpos_x), gridpos_y(gridpos_y), gridpos_z(gridpos_z), 
        globalIndex(globalIndex) {}
        
    };



};

} // namespace MeldPlugin

#endif /*OPENMM_MELD_FORCE_H_*/
