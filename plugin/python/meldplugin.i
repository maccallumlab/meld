/*
   Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
   All rights reserved
*/

%module meldplugin

%import(module="simtk.openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"


%include "std_vector.i"

namespace std {
  %template(vectord) vector<double>;
  %template(vectori) vector<int>;
};


%{
#include "MeldForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
#include <vector>
%}

/* include version information */
%pythoncode %{
__version__ = '0.6.1'
%}

/*
  The actual routines to wrap are below
*/
namespace MeldPlugin {

    class MeldForce : public OpenMM::Force {
    public:
        MeldForce(int numRDCAlignments, float rdcScaleFactor);

        void updateParametersInContext(OpenMM::Context& context);

        int getNumDistRestraints() const;

        int getNumTorsionRestraints() const;

        int getNumDistProfileRestraints() const;

        int getNumDistProfileRestParams() const;

        int getNumTorsProfileRestraints() const;

        int getNumTorsProfileRestParams() const;

        int getNumGMMRestraints() const;

        int getNumTotalRestraints() const;

        int getNumGroups() const;

        int getNumCollections() const;

        %apply int& OUTPUT {int& particle1};
        %apply int& OUTPUT {int& particle2};
        %apply int& OUTPUT {int& alignment};
        %apply float& OUTPUT {float& kappa};
        %apply float& OUTPUT {float& obs};
        %apply float& OUTPUT {float& tol};
        %apply float& OUTPUT {float& quad_cut};
        %apply float& OUTPUT {float& force_constant};
        %apply int& OUTPUT {int& globalIndex};
        void getRDCRestraintParameters(int index, int& particle1, int& particle2, int& alignment,
                                        float& kappa, float& obs, float& tol, float& quad_cut, float& force_constant,
                                        int& globalIndex) const;
        %clear int& particle1;
        %clear int& particle2;
        %clear int& alignment;
        %clear float& kappa;
        %clear float& obs;
        %clear float& tol;
        %clear float& quad_cut;
        %clear float& force_constant;
        %clear int& globalIndex;

        %apply int& OUTPUT {int& atom1};
        %apply int& OUTPUT {int& atom2};
        %apply float& OUTPUT {float& r1};
        %apply float& OUTPUT {float& r2};
        %apply float& OUTPUT {float& r3};
        %apply float& OUTPUT {float& r4};
        %apply float& OUTPUT {float& forceConstant};
        %apply int& OUTPUT {int& globalIndex};
        void getDistanceRestraintParams(int index, int& atom1, int& atom2, float& r1, float& r2, float& r3,
                float& r4, float& forceConstant, int& globalIndex) const;
        %clear int& atom1;
        %clear int& atom2;
        %clear float& r1;
        %clear float& r2;
        %clear float& r3;
        %clear float& r4;
        %clear float& forceConstant;
        %clear int& globalIndex;

        %apply int& OUTPUT {int& atom1};
        %apply int& OUTPUT {int& atom2};
        %apply int& OUTPUT {int& atom3};
        %apply int& OUTPUT {int& atom4};
        %apply float& OUTPUT {float& phi};
        %apply float& OUTPUT {float& deltaPhi};
        %apply float& OUTPUT {float& forceConstant};
        %apply int& OUTPUT {int& globalIndex};
        void getTorsionRestraintParams(int index, int& atom1, int& atom2, int& atom3, int&atom4,
                float& phi, float& deltaPhi, float& forceConstant, int& globalIndex) const;
        %clear int& atom1;
        %clear int& atom2;
        %clear int& atom3;
        %clear int& atom4;
        %clear float& phi;
        %clear float& deltaPhi;
        %clear float& forceConstant;
        %clear int& globalIndex;

        %apply int& OUTPUT {int& atom1};
        %apply int& OUTPUT {int& atom2};
        %apply float& OUTPUT {float& rMin};
        %apply float& OUTPUT {float& rMax};
        %apply int& OUTPUT {int& nBins};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a0};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a1};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a2};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a3};
        %apply float& OUTPUT {float& scaleFactor};
        %apply int& OUTPUT {int& globalIndex};
        void getDistProfileRestraintParams(int index, int& atom1, int& atom2, float& rMin, float & rMax,
                int& nBins, std::vector<double>& a0, std::vector<double>& a1, std::vector<double>& a2,
                std::vector<double>& a3, float& scaleFactor, int& globalIndex) const;
        %clear int& atom1;
        %clear int& atom2;
        %clear float& rMin;
        %clear float& rMax;
        %clear int& nBins;
        %clear vector<double>& a0;
        %clear std::vector<double>& a1;
        %clear std::vector<double>& a2;
        %clear std::vector<double>& a3;
        %clear float& scaleFactor;
        %clear int& globalIndex;

        %apply int& OUTPUT {int& atom1};
        %apply int& OUTPUT {int& atom2};
        %apply int& OUTPUT {int& atom3};
        %apply int& OUTPUT {int& atom4};
        %apply int& OUTPUT {int& atom5};
        %apply int& OUTPUT {int& atom6};
        %apply int& OUTPUT {int& atom7};
        %apply int& OUTPUT {int& atom8};
        %apply int& OUTPUT {int& nBins};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a0};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a1};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a2};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a3};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a4};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a5};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a6};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a7};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a8};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a9};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a10};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a11};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a12};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a13};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a14};
        %apply std::vector<double>& OUTPUT {std::vector<double>& a15};
        %apply float& OUTPUT {float& scaleFactor};
        %apply int& OUTPUT {int& globalIndex};
        void getTorsProfileRestraintParams(int index, int& atom1, int& atom2, int& atom3, int& atom4,
                int& atom5, int& atom6, int& atom7, int& atom8, int& nBins,
                std::vector<double>&  a0, std::vector<double>&  a1, std::vector<double>&  a2,
                std::vector<double>&  a3, std::vector<double>&  a4, std::vector<double>&  a5,
                std::vector<double>&  a6, std::vector<double>&  a7, std::vector<double>&  a8,
                std::vector<double>&  a9, std::vector<double>& a10, std::vector<double>& a11,
                std::vector<double>& a12, std::vector<double>& a13, std::vector<double>& a14,
                std::vector<double>& a15, float& scaleFactor, int& globalIndex) const;
        %clear int& atom1;
        %clear int& atom2;
        %clear int& atom3;
        %clear int& atom4;
        %clear int& atom5;
        %clear int& atom6;
        %clear int& atom7;
        %clear int& atom8;
        %clear int& nBins;
        %clear std::vector<double>& a0;
        %clear std::vector<double>& a1;
        %clear std::vector<double>& a2;
        %clear std::vector<double>& a3;
        %clear std::vector<double>& a4;
        %clear std::vector<double>& a5;
        %clear std::vector<double>& a6;
        %clear std::vector<double>& a7;
        %clear std::vector<double>& a8;
        %clear std::vector<double>& a9;
        %clear std::vector<double>& a10;
        %clear std::vector<double>& a11;
        %clear std::vector<double>& a12;
        %clear std::vector<double>& a13;
        %clear std::vector<double>& a14;
        %clear std::vector<double>& a15;
        %clear float& scaleFactor;
        %clear int& globalIndex;


        %apply int& OUTPUT {int& nPairs};
        %apply int& OUTPUT {int& nComponents};
        %apply std::vector<int>& OUTPUT {std::vector<int>& atomIndices};
        %apply std::vector<double>& OUTPUT {std::vector<double>& weights};
        %apply std::vector<double>& OUTPUT {std::vector<double>& means};
        %apply std::vector<double>& OUTPUT {std::vector<double>& precisionOnDiagonal};
        %apply std::vector<double>& OUTPUT {std::vector<double>& precisionOffDiagonal};
        %apply int& OUTPUT {int& globalIndex};
        void getGMMRestraintParams(int index, int& nPairs, int& nComponents, float& scale,
                                   std::vector<int>& atomIndices,
                                   std::vector<double>& weights,
                                   std::vector<double>& means,
                                   std::vector<double>& precisionOnDiagonal,
                                   std::vector<double>& precisionOffDiagonal,
                                   int& globalIndex) const;
        %clear int& nPairs;
        %clear int& nComponents;
        %clear int& globalIndex;
        %clear std::vector<int>& atomIndices;
        %clear std::vector<double>& weights;
        %clear std::vector<double>& means;
        %clear std::vector<double>& precisionOnDiagonal;
        %clear std::vector<double>& precisionOffDiagonal;

        %apply std::vector<int>& OUTPUT {std::vector<int>& indices};
        %apply int& OUTPUT {int& numActive};
        void getGroupParams(int index, std::vector<int>& indices, int& numActive) const;
        %clear std::vector<int>& indices;
        %clear int& numActive;

        %apply std::vector<int>& OUTPUT {std::vector<int>& indices};
        %apply int& OUTPUT {int& numActive};
        void getCollectionParams(int index, std::vector<int>& indices, int& numActive) const;
        %clear std::vector<int>& indices;
        %clear int& numActive;

        int addRDCRestraint(int particle1, int particle2, int alignment,
                            float kappa, float obs, float tol, float quad_cut, float force_constant);

        void modifyRDCRestraint(int index, int particle1, int particle2, int alignment,
                                float kappa, float obs, float tol, float quad_cut, float force_constant);

        int addDistanceRestraint(int particle1, int particle2, float r1, float r2, float r3, float r4,
                float force_constant);

        void modifyDistanceRestraint(int index, int particle1, int particle2, float r1, float r2, float r3,
                float r4, float force_constant);

        int addHyperbolicDistanceRestraint(int particle1, int particle2, float r1, float r2, float r3, float r4,
                float force_constant, float asymptote);

        void modifyHyperbolicDistanceRestraint(int index, int particle1, int particle2, float r1, float r2, float r3,
                float r4, float force_constant, float asymptote);

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

        int addGMMRestraint(int nPairs, int nComponents, float scale,
                            std::vector<int> atomIndices,
                            std::vector<double> weights,
                            std::vector<double> means,
                            std::vector<double> precisionOnDiagonal,
                            std::vector<double> precisionOffDiagonal);

        void modifyGMMRestraint(int index, int nPairs, int nComponents, float scale,
                                std::vector<int> atomIndices,
                                std::vector<double> weights,
                                std::vector<double> means,
                                std::vector<double> precisionOnDiagonal,
                                std::vector<double> precisionOffDiagonal);

        int addGroup(std::vector<int> restraint_indices, int n_active);
        void modifyGroupNumActive(int index, int numActive);

        int addCollection(std::vector<int> group_indices, int n_active);
        void modifyCollectionNumActive(int index, int numActive);
    };
}
