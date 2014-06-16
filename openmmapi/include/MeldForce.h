#ifndef OPENMM_MELD_FORCE_H_
#define OPENMM_MELD_FORCE_H_

#include "openmm/Force.h"
#include "openmm/Vec3.h"
#include "internal/windowsExportMeld.h"
#include <map>
#include <vector>

namespace MeldPlugin {

class OPENMM_EXPORT_MELD MeldForce : public OpenMM::Force {

public:
    /**
     * Default constructor
     */
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

protected:
    OpenMM::ForceImpl* createImpl() const;

private:
    class TorsionRestraintInfo;
    class DistanceRestraintInfo;
    class DistProfileRestraintInfo;
    class TorsProfileRestraintInfo;
    class GroupInfo;
    class CollectionInfo;
    int n_restraints;
    std::vector<DistanceRestraintInfo> distanceRestraints;
    std::vector<TorsionRestraintInfo> torsions;
    std::vector<DistProfileRestraintInfo> distProfileRestraints;
    std::vector<TorsProfileRestraintInfo> torsProfileRestraints;
    std::vector<GroupInfo> groups;
    std::vector<CollectionInfo> collections;

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
};

} // namespace MeldPlugin

#endif /*OPENMM_MELD_FORCE_H_*/
