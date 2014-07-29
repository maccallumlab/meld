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
import simtk.openmm as mm
import simtk.unit as unit
%}


/* strip the units off of all input arguments */
%pythonprepend %{
try:
    args=mm.stripUnits(args)
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

