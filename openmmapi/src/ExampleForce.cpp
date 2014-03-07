/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2014 Stanford University and the Authors.           *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "ExampleForce.h"
#include "internal/ExampleForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"

using namespace ExamplePlugin;
using namespace OpenMM;
using namespace std;

ExampleForce::ExampleForce() {
}

int ExampleForce::addBond(int particle1, int particle2, double length, double k) {
    bonds.push_back(BondInfo(particle1, particle2, length, k));
    return bonds.size()-1;
}

void ExampleForce::getBondParameters(int index, int& particle1, int& particle2, double& length, double& k) const {
    ASSERT_VALID_INDEX(index, bonds);
    particle1 = bonds[index].particle1;
    particle2 = bonds[index].particle2;
    length = bonds[index].length;
    k = bonds[index].k;
}

void ExampleForce::setBondParameters(int index, int particle1, int particle2, double length, double k) {
    ASSERT_VALID_INDEX(index, bonds);
    bonds[index].particle1 = particle1;
    bonds[index].particle2 = particle2;
    bonds[index].length = length;
    bonds[index].k = k;
}

ForceImpl* ExampleForce::createImpl() const {
    return new ExampleForceImpl(*this);
}

void ExampleForce::updateParametersInContext(Context& context) {
    dynamic_cast<ExampleForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}
