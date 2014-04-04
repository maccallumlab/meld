%module exampleplugin

%import(module="simtk.openmm") "OpenMMSwigHeaders.i"

%{
#include "ExampleForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
%}

namespace ExamplePlugin {

class ExampleForce : public OpenMM::Force {
public:
    ExampleForce();
    int getNumBonds() const;
    int addBond(int particle1, int particle2, double length, double k);
    void getBondParameters(int index, int& particle1, int& particle2, double& length, double& k) const;
    void setBondParameters(int index, int particle1, int particle2, double length, double k);
    void updateParametersInContext(OpenMM::Context& context);
};

}

