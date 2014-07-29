%module exampleplugin

%import(module="simtk.openmm") "OpenMMSwigHeaders.i"


/*
 * The following lines are needed to handle std::vector.
 * Similar lines may be needed for vectors of vectors or
 * for other STL types like maps.
 */

%include "std_vector.i"
namespace std {
  %template(vectord) vector<double>;
  %template(vectori) vector<int>;
};

%{
#include "ExampleForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
%}


/*
 * The code below strips all units before the wrapper
 * functions are called. This code also converts numpy
 * arrays to lists.
*/

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
 * Add units to function outputs.
*/
%pythonappend ExamplePlugin::ExampleForce::getBondParameters(int index, int& particle1, int& particle2,
                                                             double& length, double& k) const %{
    val[2] = unit.Quantity(val[2], unit.nanometer)
    val[3] = unit.Quantity(val[3], unit.kilojoule_per_mole / (unit.nanometer * unit.nanometer))
%}


namespace ExamplePlugin {

class ExampleForce : public OpenMM::Force {
public:
    ExampleForce();

    int getNumBonds() const;

    int addBond(int particle1, int particle2, double length, double k);

    void setBondParameters(int index, int particle1, int particle2, double length, double k);

    void updateParametersInContext(OpenMM::Context& context);

    /*
     * The reference parameters to this function are output values.
     * Marking them as such will cause swig to return a tuple.
    */
    %apply int& OUTPUT {int& particle1};
    %apply int& OUTPUT {int& particle2};
    %apply double& OUTPUT {double& length};
    %apply double& OUTPUT {double& k};
    void getBondParameters(int index, int& particle1, int& particle2, double& length, double& k) const;
    %clear int& particle1;
    %clear int& particle2;
    %clear double& length;
    %clear double& k;
};

}

