#ifndef OPENMM_RDC_FORCE_H_
#define OPENMM_RDC_FORCE_H_

#include "openmm/Force.h"
#include "openmm/Vec3.h"
#include "internal/windowsExportMeld.h"
#include <map>
#include <vector>

namespace MeldPlugin {

class OPENMM_EXPORT_MELD RdcForce : public OpenMM::Force {

public:
    /**
     * Default constructor
     */
    RdcForce();

    void updateParametersInContext(OpenMM::Context& context);
    int getNumExperiments() const;
    int getNumRestraints(int experiment) const;
    int getNumTotalRestraints() const;

    int addExperiment(std::vector<int> rdcIndices);
    int addRdcRestraint(int particle1, int particle2, float kappa, float dObs, float tolerance,
            float force_const, float weight);
    void updateRdcRestraint(int index, int particle1, int particle2, float kappa, float dObs,
            float tolerance, float force_const, float weight);

    void getExperimentInfo(int index, std::vector<int>& restraints) const;
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
