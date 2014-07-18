%module meldplugin

%import(module="simtk.openmm") "OpenMMSwigHeaders.i"

%{
#include "MeldForce.h"
#include "RdcForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include <vector>
%}

namespace MeldPlugin {

    class MeldForce : public OpenMM::Force {
    public:
        MeldForce();
        void updateParametersInContext(OpenMM::Context& context);
        int getNumDistRestraints() const;
        int getNumTorsionRestraints() const;
        int getNumDistProfileRestraints() const;
        int getNumDistProfileRestParams() const;
        int getNumTorsProfileRestraints() const;
        int getNumTorsProfileRestParams() const;
        int getNumTotalRestraints() const;
        int getNumGroups() const;
        int getNumCollections() const;
        void getDistanceRestraintParams(int index, int& atom1, int& atom2, float& r1, float& r2, float& r3,
                float& r4, float& forceConstant, int& globalIndex) const;
        void getTorsionRestraintParams(int index, int& atom1, int& atom2, int& atom3, int&atom4,
                float& phi, float& deltaPhi, float& forceConstant, int& globalIndex) const;
        void getDistProfileRestraintParams(int index, int& atom1, int& atom2, float& rMin, float & rMax,
                int& nBins, std::vector<double>& a0, std::vector<double>& a1, std::vector<double>& a2,
                std::vector<double>& a3, float& scaleFactor, int& globalIndex) const;
        void getTorsProfileRestraintParams(int index, int& atom1, int& atom2, int& atom3, int& atom4,
                int& atom5, int& atom6, int& atom7, int& atom8, int& nBins,
                std::vector<double>&  a0, std::vector<double>&  a1, std::vector<double>&  a2,
                std::vector<double>&  a3, std::vector<double>&  a4, std::vector<double>&  a5,
                std::vector<double>&  a6, std::vector<double>&  a7, std::vector<double>&  a8,
                std::vector<double>&  a9, std::vector<double>& a10, std::vector<double>& a11,
                std::vector<double>& a12, std::vector<double>& a13, std::vector<double>& a14,
                std::vector<double>& a15, float& scaleFactor, int& globalIndex) const;
        void getGroupParams(int index, std::vector<int>& indices, int& numActive) const;
        void getCollectionParams(int index, std::vector<int>& indices, int& numActive) const;
        int addDistanceRestraint(int particle1, int particle2, float r1, float r2, float r3, float r4,
                float force_constant);
        void modifyDistanceRestraint(int index, int particle1, int particle2, float r1, float r2, float r3,
                float r4, float force_constant);
        int addTorsionRestraint(int atom1, int atom2, int atom3, int atom4, float phi, float deltaPhi, float forceConstant);
        void modifyTorsionRestraint(int index, int atom1, int atom2, int atom3, int atom4, float phi,
                float deltaPhi, float forceConstant);
        int addDistProfileRestraint(int atom1, int atom2, float rMin, float rMax, int nBins, std::vector<double> a0,
                std::vector<double> a1, std::vector<double> a2, std::vector<double> a3, float scaleFactor);
        void modifyDistProfileRestraint(int index, int atom1, int atom2, float rMin, float rMax, int nBins,
                std::vector<double> a0, std::vector<double> a1, std::vector<double> a2, std::vector<double> a3,
                float scaleFactor);
        int addTorsProfileRestraint(int atom1, int atom2, int atom3, int atom4,
                int atom5, int atom6, int atom7, int atom8, int nBins,
                std::vector<double>  a0, std::vector<double>  a1, std::vector<double>  a2,
                std::vector<double>  a3, std::vector<double>  a4, std::vector<double>  a5,
                std::vector<double>  a6, std::vector<double>  a7, std::vector<double>  a8,
                std::vector<double>  a9, std::vector<double> a10, std::vector<double> a11,
                std::vector<double> a12, std::vector<double> a13, std::vector<double> a14,
                std::vector<double> a15, float scaleFactor);
        void modifyTorsProfileRestraint(int index, int atom1, int atom2, int atom3, int atom4,
                int atom5, int atom6, int atom7, int atom8, int nBins,
                std::vector<double>  a0, std::vector<double>  a1, std::vector<double>  a2,
                std::vector<double>  a3, std::vector<double>  a4, std::vector<double>  a5,
                std::vector<double>  a6, std::vector<double>  a7, std::vector<double>  a8,
                std::vector<double>  a9, std::vector<double> a10, std::vector<double> a11,
                std::vector<double> a12, std::vector<double> a13, std::vector<double> a14,
                std::vector<double> a15, float scaleFactor);
        int addGroup(std::vector<int> restraint_indices, int n_active);
        int addCollection(std::vector<int> group_indices, int n_active);
    };

    class RdcForce : public OpenMM::Force {
    public:
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
    };
}

