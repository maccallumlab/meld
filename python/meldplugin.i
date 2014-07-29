%module meldplugin

%import(module="simtk.openmm") "OpenMMSwigHeaders.i"


%include "std_vector.i"

namespace std {
  %template(vectord) vector<double>;
  %template(vectori) vector<int>;
};


%{
#include "MeldForce.h"
#include "RdcForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include <vector>
%}


/* python code for routines to strip units */
%pythoncode %{

import numpy

import sys
import math
import simtk.unit as unit
from simtk.openmm.vec3 import Vec3


# Strings can cause trouble
# as can any container that has infinite levels of containment
def _is_string(x):
    # step 1) String is always a container
    # and its contents are themselves containers.
    try:
        first_item = iter(x).next()
        inner_item = iter(first_item).next()
        if first_item == inner_item:
            return True
        else:
            return False
    except TypeError:
        return False
    except StopIteration:
        return False
    except ValueError:
        return False


def stripUnits(args):
    """
    getState(self, quantity) 
          -> value with *no* units

    Examples
    >>> import simtk

    >>> x = 5
    >>> print x
    5

    >>> x = stripUnits((5*simtk.unit.nanometer,))
    >>> x
    (5,)

    >>> arg1 = 5*simtk.unit.angstrom
    >>> x = stripUnits((arg1,))
    >>> x
    (0.5,)

    >>> arg1 = 5
    >>> x = stripUnits((arg1,))
    >>> x
    (5,)

    >>> arg1 = (1*simtk.unit.angstrom, 5*simtk.unit.angstrom)
    >>> x = stripUnits((arg1,))
    >>> x
    ((0.10000000000000001, 0.5),)

    >>> arg1 = (1*simtk.unit.angstrom,
    ...         5*simtk.unit.kilojoule_per_mole,
    ...         1*simtk.unit.kilocalorie_per_mole)
    >>> y = stripUnits((arg1,))
    >>> y
    ((0.10000000000000001, 5, 4.1840000000000002),)

    """
    newArgList=[]
    for arg in args:
        if 'numpy' in sys.modules and isinstance(arg, numpy.ndarray):
           arg = arg.tolist()
        elif unit.is_quantity(arg):
            # JDC: Ugly workaround for OpenMM using 'bar' for fundamental pressure unit.
            if arg.unit.is_compatible(unit.bar):
                arg = arg / unit.bar
            else:
                arg=arg.value_in_unit_system(unit.md_unit_system)                
            # JDC: End workaround.
        elif isinstance(arg, dict):
            newKeys = stripUnits(arg.keys())
            newValues = stripUnits(arg.values())
            arg = dict(zip(newKeys, newValues))
        elif not _is_string(arg):
            try:
                iter(arg)
                # Reclusively strip units from all quantities
                arg=stripUnits(arg)
            except TypeError:
                pass
        newArgList.append(arg)
    return tuple(newArgList)
%}

%pythonappend OpenMM::Context::Context %{
    self._system = args[0]
    self._integrator = args[1]
%}

/* strip the units off of all input arguments */
%pythonprepend %{
try:
    args=stripUnits(args)
except UnboundLocalError:
    pass
%}


/*
  The actual routines to wrap are below
*/
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

