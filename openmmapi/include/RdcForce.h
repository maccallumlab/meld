/*
   Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
   All rights reserved
*/

#ifndef OPENMM_RDC_FORCE_H_
#define OPENMM_RDC_FORCE_H_

#include "openmm/Force.h"
#include "openmm/Vec3.h"
#include "internal/windowsExportMeld.h"
#include <map>
#include <vector>

namespace MeldPlugin {

/**
 * This is the RDC Force.
 */

class OPENMM_EXPORT_MELD RdcForce : public OpenMM::Force {

public:
    /**
     * Default constructor
     */
    RdcForce();

    /**
     * Update the per-restraint parameters in a Context to match those stored in this Force object.  This method provides
     * an efficient method to update certain parameters in an existing Context without needing to reinitialize it.
     * Simply call updateRdcRestaint() to modify the parameters of a restraint, then call updateParametersInContext()
     * to copy them over to the Context.
     * 
     * This method has several limitations.  The only information it updates is the values of per-restraint parameters.
     * All other aspects of the Force (such as the energy function) are unaffected and can only be changed by reinitializing
     * the Context.  The set of particles involved in a restraint cannot be changed, nor can new restraints be added.
     */
    void updateParametersInContext(OpenMM::Context& context);

    /**
     * @return The number of experiments.
     */
    int getNumExperiments() const;

    /**
     * @param experiment  The index of the experiment.
     * @return The number of RDC restraints.
     */
    int getNumRestraints(int experiment) const;

    /**
     * @return The number of RDC restraints.
     */
    int getNumTotalRestraints() const;

    /**
     * Create a new experiment.
     *
     * @param rdcIndices  The indices of the Rdc restraints in this experiment.
     * @return The index of the new experiment.
     */
    int addExperiment(std::vector<int> rdcIndices);

    /**
     * Create new RDC restraint.
     *
     * @param particle1  The index of atom 1
     * @param particle2  The index of atom 2
     * @param kappa  The kappa parameter
     * @param dObs  The dObs parameter
     * @param tolerance  The tolerance parameter
     * @param force_const  The force constant of the restraint
     * @param weight  The weight parameter
     * @return index  The index of the new restraint
     */
    int addRdcRestraint(int particle1, int particle2, float kappa, float dObs, float tolerance,
            float force_const, float weight);

    /**
     * Modify an existing RDC restraint.
     *
     * @param index  The index of the restraint
     * @param particle1  The index of atom 1
     * @param particle2  The index of atom 2
     * @param kappa  The kappa parameter
     * @param dObs  The dObs parameter
     * @param tolerance  The tolerance parameter
     * @param force_const  The force constant of the restraint
     * @param weight  The weight parameter
     */
    void updateRdcRestraint(int index, int particle1, int particle2, float kappa, float dObs,
            float tolerance, float force_const, float weight);

    /**
     * Get the parameters of an experiment.
     *
     * @param index  The index of the experiment
     * @param restraints  The indices of the restraints in the experiment.
     */
    void getExperimentInfo(int index, std::vector<int>& restraints) const;

    /**
     * Get the parameters of an RDC restraint.
     *
     * @param index  The index of the restraint
     * @param particle1  The index of atom 1
     * @param particle2  The index of atom 2
     * @param kappa  The kappa parameter
     * @param dObs  The dObs parameter
     * @param tolerance  The tolerance parameter
     * @param force_const  The force constant of the restraint
     * @param weight  The weight parameter
     * @param globalIndex  The global index of the restraint
     */
    void getRdcRestraintInfo(int index, int& particle1, int& partcile2, float& kappa,
            float& dObs, float& tolerance, float& force_const, float& weight,
            int& globalIndex) const;

protected:
    OpenMM::ForceImpl* createImpl() const;

private:
    class RdcRestraintInfo;
    class ExperimentInfo;
    int numRestraints;
    std::vector<RdcRestraintInfo> rdcRestraints;
    std::vector<ExperimentInfo> experiments;

    class RdcRestraintInfo {
    public:
        int atom1, atom2;
        float kappa, dObs, tolerance, force_const, weight;
        int globalIndex;

        RdcRestraintInfo() {
            atom1 = atom2 = -1;
            kappa = 0;
            dObs = 0;
            tolerance = 0.;
            force_const = 0.;
            weight = 0.;
            globalIndex =  -1;
        }

        RdcRestraintInfo(int atom1, int atom2, float kappa, float dObs, float tolerance,
                float force_const, float weight, int globalIndex) :
            atom1(atom1), atom2(atom2), kappa(kappa), dObs(dObs), tolerance(tolerance),
            force_const(force_const), weight(weight), globalIndex(globalIndex) {
        }
    };

    class ExperimentInfo {
    public:
        std::vector<int> rdcIndices;

        ExperimentInfo() {
        }

        ExperimentInfo(std::vector<int> rdcIndices) :
            rdcIndices(rdcIndices) {
        }
    };
};

} // namespace MeldPlugin

#endif /*OPENMM_RDC_FORCE_H_*/
