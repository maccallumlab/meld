#include "RdcForceProxy.h"
#include "RdcForce.h"
#include "openmm/serialization/SerializationNode.h"
#include <sstream>

using namespace MeldPlugin;
using namespace OpenMM;
using namespace std;

RdcForceProxy::RdcForceProxy() : SerializationProxy("RdcForce") {
}

void RdcForceProxy::serialize(const void* object, SerializationNode& node) const {
    node.setIntProperty("version", 1);
    const RdcForce& force = *reinterpret_cast<const RdcForce*>(object);
    
    SerializationNode& rdcRestraints = node.createChildNode("RdcRestraints");
    for (int i = 0; i < force.getNumTotalRestraints(); i++) {
        int particle1, particle2, globalIndex;
        float kappa, dObs, tolerance, force_const, weight;
        force.getRdcRestraintInfo(i, particle1, particle2, kappa, dObs, tolerance, force_const, weight, globalIndex);
        SerializationNode& rdc = rdcRestraints.createChildNode("RdcRestraint");
        rdc.setIntProperty("particle1", particle1);
        rdc.setIntProperty("particle2", particle2);
        rdc.setDoubleProperty("kappa", kappa);
        rdc.setDoubleProperty("dObs", dObs);
        rdc.setDoubleProperty("tolerance", tolerance);
        rdc.setDoubleProperty("force_const", force_const);
        rdc.setDoubleProperty("weight", weight);
        rdc.setIntProperty("globalIndex", globalIndex);
    }

    SerializationNode& experiments = node.createChildNode("Experiments");
    for (int i = 0; i < force.getNumExperiments(); i++) {
        std::vector<int> indices;
        force.getExperimentInfo(i, indices);
        SerializationNode& expmt = experiments.createChildNode("Experiment");
        SerializationNode& restraintIndices = expmt.createChildNode("RestraintIndices");
        for (int j = 0; j < indices.size(); j++) {
            SerializationNode& restraintIndex = restraintIndices.createChildNode("RestraintIndex");
            restraintIndex.setIntProperty("index", indices[j]);
        }
    }
}

void* RdcForceProxy::deserialize(const SerializationNode& node) const {
    if (node.getIntProperty("version") != 1)
        throw OpenMMException("Unsupported version number");
    RdcForce* force = new RdcForce();
    try {
        const SerializationNode& rdcRestraints = node.getChildNode("RdcRestraints");
        for (int i = 0; i < (int) rdcRestraints.getChildren().size(); i++) {
            const SerializationNode& dr = rdcRestraints.getChildren()[i];
            int particle1 = dr.getIntProperty("particle1");
            int particle2 = dr.getIntProperty("particle2");
            float kappa = dr.getDoubleProperty("kappa");
            float dObs = dr.getDoubleProperty("dObs");
            float tolerance = dr.getDoubleProperty("tolerance");
            float force_const = dr.getDoubleProperty("force_const");
            float weight = dr.getDoubleProperty("weight");
            force->addRdcRestraint(particle1, particle2, kappa, dObs, tolerance, force_const, weight);
        }

        const SerializationNode& experiments = node.getChildNode("Experiments");
        for (int i = 0; i < (int) experiments.getChildren().size(); i++) {
            const SerializationNode& expmt = experiments.getChildren()[i];
            const SerializationNode& restraintIndices = expmt.getChildNode("RestraintIndices");
            int n = restraintIndices.getChildren().size();
            std::vector<int> indices(n);
            for (int j = 0; j < n; j++) {
                indices[j] = restraintIndices.getChildren()[j].getIntProperty("index");
            }
            force->addExperiment(indices);
        }
    }
    catch (...) {
        delete force;
        throw;
    }
    return force;
}
