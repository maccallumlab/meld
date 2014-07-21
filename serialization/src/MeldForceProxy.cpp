#include "MeldForceProxy.h"
#include "MeldForce.h"
#include "openmm/serialization/SerializationNode.h"
#include <sstream>

using namespace MeldPlugin;
using namespace OpenMM;
using namespace std;

MeldForceProxy::MeldForceProxy() : SerializationProxy("MeldForce") {
}

void MeldForceProxy::serialize(const void* object, SerializationNode& node) const {
    node.setIntProperty("version", 1);
    const MeldForce& force = *reinterpret_cast<const MeldForce*>(object);
    
    SerializationNode& distanceRestraints = node.createChildNode("DistanceRestraints");
    for (int i = 0; i < force.getNumDistRestraints(); i++) {
        int atom1, atom2, globalIndex;
        float r1, r2, r3, r4, forceConstant;
        force.getDistanceRestraintParams(i, atom1, atom2, r1, r2, r3, r4, forceConstant, globalIndex);
        SerializationNode& dr = distanceRestraints.createChildNode("DistanceRestraint");
        dr.setIntProperty("atom1", atom1);
        dr.setIntProperty("atom2", atom2);
        dr.setDoubleProperty("r1", r1);
        dr.setDoubleProperty("r2", r2);
        dr.setDoubleProperty("r3", r3);
        dr.setDoubleProperty("r4", r4);
        dr.setDoubleProperty("forceConstant", forceConstant);
        dr.setIntProperty("globalIndex", globalIndex);
    }

    SerializationNode& groups = node.createChildNode("Groups");
    for (int i = 0; i < force.getNumGroups(); i++) {
        std::vector<int> indices;
        int numActive;
        force.getGroupParams(i, indices, numActive);
        SerializationNode& group = groups.createChildNode("Group");
        group.setIntProperty("numActive", numActive);
        SerializationNode& restraintIndices = group.createChildNode("RestraintIndices");
        for (int j = 0; j < indices.size(); j++) {
            SerializationNode& restraintIndex = restraintIndices.createChildNode("RestraintIndex");
            restraintIndex.setIntProperty("index", indices[j]);
        }
    }

    SerializationNode& collections = node.createChildNode("Collections");
    for (int i = 0; i < force.getNumCollections(); i++) {
        std::vector<int> indices;
        int numActive;
        force.getCollectionParams(i, indices, numActive);
        SerializationNode& collection = collections.createChildNode("Collection");
        collection.setIntProperty("numActive", numActive);
        SerializationNode& groupIndices = collection.createChildNode("GroupIndices");
        for (int j = 0; j < indices.size(); j++) {
            SerializationNode& groupIndex = groupIndices.createChildNode("groupIndex");
            groupIndex.setIntProperty("index", indices[j]);
        }
    }
}

void* MeldForceProxy::deserialize(const SerializationNode& node) const {
    if (node.getIntProperty("version") != 1)
        throw OpenMMException("Unsupported version number");
    MeldForce* force = new MeldForce();
    try {
        const SerializationNode& distanceRestraints = node.getChildNode("DistanceRestraints");
        for (int i = 0; i < (int) distanceRestraints.getChildren().size(); i++) {
            const SerializationNode& dr = distanceRestraints.getChildren()[i];
            int atom1 = dr.getIntProperty("atom1");
            int atom2 = dr.getIntProperty("atom2");
            float r1 = dr.getDoubleProperty("r1");
            float r2 = dr.getDoubleProperty("r2");
            float r3 = dr.getDoubleProperty("r3");
            float r4 = dr.getDoubleProperty("r4");
            float forceConstant = dr.getDoubleProperty("forceConstant");
            force->addDistanceRestraint(atom1, atom2, r1, r2, r3, r4, forceConstant);
        }

        const SerializationNode& groups = node.getChildNode("Groups");
        for (int i = 0; i < (int) groups.getChildren().size(); i++) {
            const SerializationNode& group = groups.getChildren()[i];
            int numActive = group.getIntProperty("numActive");
            const SerializationNode& restraintIndices = group.getChildNode("RestraintIndices");
            int n = restraintIndices.getChildren().size();
            std::vector<int> indices(n);
            for (int j = 0; j < n; j++) {
                indices[j] = restraintIndices.getChildren()[j].getIntProperty("index");
            }
            force->addGroup(indices, numActive);
        }

        const SerializationNode& collections = node.getChildNode("Collections");
        for (int i = 0; i < (int) collections.getChildren().size(); i++) {
            const SerializationNode& collection = collections.getChildren()[i];
            int numActive = collection.getIntProperty("numActive");
            const SerializationNode& groupIndices = collection.getChildNode("GroupIndices");
            int n = groupIndices.getChildren().size();
            std::vector<int> indices(n);
            for (int j = 0; j < n; j++) {
                indices[j] = groupIndices.getChildren()[j].getIntProperty("index");
            }
            force->addCollection(indices, numActive);
        }
    }
    catch (...) {
        delete force;
        throw;
    }
    return force;
}
