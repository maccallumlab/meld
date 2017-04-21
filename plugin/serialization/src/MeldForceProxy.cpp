/*
   Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
   All rights reserved
*/

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

    // serialize distance restraints
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

    // serialize torsion restraints
    SerializationNode& torsionRestraints = node.createChildNode("TorsionRestraints");
    for (int i = 0; i < force.getNumTorsionRestraints(); i++) {
        int atom1, atom2, atom3, atom4, globalIndex;
        float phi, deltaPhi, forceConstant;
        force.getTorsionRestraintParams(i, atom1, atom2, atom3, atom4, phi, deltaPhi, forceConstant, globalIndex);
        SerializationNode& tr = torsionRestraints.createChildNode("TorsionRestraint");
        tr.setIntProperty("atom1", atom1);
        tr.setIntProperty("atom2", atom2);
        tr.setIntProperty("atom3", atom3);
        tr.setIntProperty("atom4", atom4);
        tr.setDoubleProperty("phi", phi);
        tr.setDoubleProperty("deltaPhi", deltaPhi);
        tr.setDoubleProperty("forceConstant", forceConstant);
        tr.setIntProperty("globalIndex", globalIndex);
    }

    // serialize distance profile restraints
    SerializationNode& distanceProfRestraints = node.createChildNode("DistanceProfRestraints");
    for (int i = 0; i < force.getNumDistProfileRestraints(); i++) {
        int atom1, atom2, nBins, globalIndex;
        float rMin, rMax, scaleFactor;
        std::vector<double> a0;
        std::vector<double> a1;
        std::vector<double> a2;
        std::vector<double> a3;
        force.getDistProfileRestraintParams(i, atom1, atom2, rMin, rMax, nBins, a0, a1, a2, a3, scaleFactor, globalIndex);
        SerializationNode& dpr = distanceProfRestraints.createChildNode("DistanceProfRestraint");
        dpr.setIntProperty("atom1", atom1);
        dpr.setIntProperty("atom2", atom2);
        dpr.setIntProperty("nBins", nBins);
        dpr.setDoubleProperty("rMin", rMin);
        dpr.setDoubleProperty("rMax", rMax);
        dpr.setDoubleProperty("scaleFactor", scaleFactor);
        dpr.setIntProperty("globalIndex", globalIndex);
        SerializationNode& a0Params = dpr.createChildNode("a0Params");
        for (int j = 0; j < a0.size(); j++) {
            SerializationNode& aParam = a0Params.createChildNode("a0Param");
            aParam.setDoubleProperty("param", a0[j]);
        }
        SerializationNode& a1Params = dpr.createChildNode("a1Params");
        for (int j = 0; j < a1.size(); j++) {
            SerializationNode& aParam = a1Params.createChildNode("a1Param");
            aParam.setDoubleProperty("param", a1[j]);
        }
        SerializationNode& a2Params = dpr.createChildNode("a2Params");
        for (int j = 0; j < a2.size(); j++) {
            SerializationNode& aParam = a2Params.createChildNode("a2Param");
            aParam.setDoubleProperty("param", a2[j]);
        }
        SerializationNode& a3Params = dpr.createChildNode("a3Params");
        for (int j = 0; j < a3.size(); j++) {
            SerializationNode& aParam = a3Params.createChildNode("a3Param");
            aParam.setDoubleProperty("param", a3[j]);
        }
    }

    // serialize torsion profile restraints
    SerializationNode& torsProfRestraints = node.createChildNode("TorsProfRestraints");
    for (int i = 0; i < force.getNumTorsProfileRestraints(); i++) {
        int atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8;
        int nBins, globalIndex;
        float scaleFactor;
        std::vector<double> a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15;
        force.getTorsProfileRestraintParams(i, atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8,
                                            nBins, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15,
                                            scaleFactor, globalIndex);
        SerializationNode& tpr = torsProfRestraints.createChildNode("TorsProfRestraint");
        tpr.setIntProperty("atom1", atom1);
        tpr.setIntProperty("atom2", atom2);
        tpr.setIntProperty("atom3", atom3);
        tpr.setIntProperty("atom4", atom4);
        tpr.setIntProperty("atom5", atom5);
        tpr.setIntProperty("atom6", atom6);
        tpr.setIntProperty("atom7", atom7);
        tpr.setIntProperty("atom8", atom8);
        tpr.setIntProperty("nBins", nBins);
        tpr.setDoubleProperty("scaleFactor", scaleFactor);
        tpr.setIntProperty("globalIndex", globalIndex);
        SerializationNode& a0Params = tpr.createChildNode("a0Params");
        for (int j = 0; j < a0.size(); j++) {
            SerializationNode& aParam = a0Params.createChildNode("a0Param");
            aParam.setDoubleProperty("param", a0[j]);
        }
        SerializationNode& a1Params = tpr.createChildNode("a1Params");
        for (int j = 0; j < a1.size(); j++) {
            SerializationNode& aParam = a1Params.createChildNode("a1Param");
            aParam.setDoubleProperty("param", a1[j]);
        }
        SerializationNode& a2Params = tpr.createChildNode("a2Params");
        for (int j = 0; j < a2.size(); j++) {
            SerializationNode& aParam = a2Params.createChildNode("a2Param");
            aParam.setDoubleProperty("param", a2[j]);
        }
        SerializationNode& a3Params = tpr.createChildNode("a3Params");
        for (int j = 0; j < a3.size(); j++) {
            SerializationNode& aParam = a3Params.createChildNode("a3Param");
            aParam.setDoubleProperty("param", a3[j]);
        }
        SerializationNode& a4Params = tpr.createChildNode("a4Params");
        for (int j = 0; j < a4.size(); j++) {
            SerializationNode& aParam = a4Params.createChildNode("a4Param");
            aParam.setDoubleProperty("param", a4[j]);
        }
        SerializationNode& a5Params = tpr.createChildNode("a5Params");
        for (int j = 0; j < a5.size(); j++) {
            SerializationNode& aParam = a5Params.createChildNode("a5Param");
            aParam.setDoubleProperty("param", a5[j]);
        }
        SerializationNode& a6Params = tpr.createChildNode("a6Params");
        for (int j = 0; j < a6.size(); j++) {
            SerializationNode& aParam = a6Params.createChildNode("a6Param");
            aParam.setDoubleProperty("param", a6[j]);
        }
        SerializationNode& a7Params = tpr.createChildNode("a7Params");
        for (int j = 0; j < a7.size(); j++) {
            SerializationNode& aParam = a7Params.createChildNode("a7Param");
            aParam.setDoubleProperty("param", a7[j]);
        }
        SerializationNode& a8Params = tpr.createChildNode("a8Params");
        for (int j = 0; j < a8.size(); j++) {
            SerializationNode& aParam = a8Params.createChildNode("a8Param");
            aParam.setDoubleProperty("param", a8[j]);
        }
        SerializationNode& a9Params = tpr.createChildNode("a9Params");
        for (int j = 0; j < a9.size(); j++) {
            SerializationNode& aParam = a9Params.createChildNode("a9Param");
            aParam.setDoubleProperty("param", a9[j]);
        }
        SerializationNode& a10Params = tpr.createChildNode("a10Params");
        for (int j = 0; j < a10.size(); j++) {
            SerializationNode& aParam = a10Params.createChildNode("a10Param");
            aParam.setDoubleProperty("param", a10[j]);
        }
        SerializationNode& a11Params = tpr.createChildNode("a11Params");
        for (int j = 0; j < a11.size(); j++) {
            SerializationNode& aParam = a11Params.createChildNode("a11Param");
            aParam.setDoubleProperty("param", a11[j]);
        }
        SerializationNode& a12Params = tpr.createChildNode("a12Params");
        for (int j = 0; j < a12.size(); j++) {
            SerializationNode& aParam = a12Params.createChildNode("a12Param");
            aParam.setDoubleProperty("param", a12[j]);
        }
        SerializationNode& a13Params = tpr.createChildNode("a13Params");
        for (int j = 0; j < a13.size(); j++) {
            SerializationNode& aParam = a13Params.createChildNode("a13Param");
            aParam.setDoubleProperty("param", a13[j]);
        }
        SerializationNode& a14Params = tpr.createChildNode("a14Params");
        for (int j = 0; j < a14.size(); j++) {
            SerializationNode& aParam = a14Params.createChildNode("a14Param");
            aParam.setDoubleProperty("param", a14[j]);
        }
        SerializationNode& a15Params = tpr.createChildNode("a15Params");
        for (int j = 0; j < a15.size(); j++) {
            SerializationNode& aParam = a15Params.createChildNode("a15Param");
            aParam.setDoubleProperty("param", a15[j]);
        }
    }

    // serialize gmm restraints
    SerializationNode& gmmRestraints = node.createChildNode("GMMRestraints");
    for (int i = 0; i < force.getNumGMMRestraints(); i++) {
        //
        int nPairs, nComponents, globalIndex;
        float scale;
        std::vector<int> atomIndices;
        std::vector<double> weights, means, precOnDiag, precOffDiag;
        force.getGMMRestraintParams(i, nPairs, nComponents, scale, atomIndices, weights,
                                    means, precOnDiag, precOffDiag, globalIndex);
        SerializationNode& gr = gmmRestraints.createChildNode("GMMRestraint");
        gr.setIntProperty("nPairs", nPairs);
        gr.setIntProperty("nComponents", nComponents);
        gr.setDoubleProperty("scale", scale);

        SerializationNode& a = gr.createChildNode("atomIndices");
        for(const auto& atom : atomIndices) {
            SerializationNode& ap = a.createChildNode("atomIndex");
            ap.setIntProperty("index", atom);
        }

        SerializationNode& w = gr.createChildNode("weights");
        for(const auto& weight : weights) {
            SerializationNode& wp = w.createChildNode("weight");
            wp.setDoubleProperty("weight", weight);
        }

        SerializationNode& m = gr.createChildNode("means");
        for(const auto& mean : means) {
            SerializationNode& mp = m.createChildNode("mean");
            mp.setDoubleProperty("mean", mean);
        }

        SerializationNode& d = gr.createChildNode("precisionOnDiagonals");
        for(const auto& diag : precOnDiag) {
            SerializationNode& dp = d.createChildNode("precisionOnDiagonal");
            dp.setDoubleProperty("prec", diag);
        }

        SerializationNode& o = gr.createChildNode("precisionOffDiagonals");
        for(const auto& off : precOffDiag) {
            SerializationNode& op = o.createChildNode("precisionOffDiagonal");
            op.setDoubleProperty("prec", off);
        }

        gr.setIntProperty("globalIndex", globalIndex);
    }

    // serialize groups
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

    // serialize collections
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
        // deserialize distance restraints
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

        // deserialize torsion restraints
        const SerializationNode& torsionRestraints = node.getChildNode("TorsionRestraints");
        for (int i = 0; i < (int) torsionRestraints.getChildren().size(); i++) {
            const SerializationNode& tr = torsionRestraints.getChildren()[i];
            int atom1 = tr.getIntProperty("atom1");
            int atom2 = tr.getIntProperty("atom2");
            int atom3 = tr.getIntProperty("atom3");
            int atom4 = tr.getIntProperty("atom4");
            float phi = tr.getDoubleProperty("phi");
            float deltaPhi = tr.getDoubleProperty("deltaPhi");
            float forceConstant = tr.getDoubleProperty("forceConstant");
            force->addTorsionRestraint(atom1, atom2, atom3, atom4, phi, deltaPhi, forceConstant);
        }

        // deserialize distance profile restraints
        const SerializationNode& distProfRestraints = node.getChildNode("DistanceProfRestraints");
        for (int i = 0; i < (int) distProfRestraints.getChildren().size(); i++) {
            const SerializationNode& dpr = distProfRestraints.getChildren()[i];
            int atom1 = dpr.getIntProperty("atom1");
            int atom2 = dpr.getIntProperty("atom2");
            int nBins = dpr.getIntProperty("nBins");
            float rMin = dpr.getDoubleProperty("rMin");
            float rMax = dpr.getDoubleProperty("rMax");
            float scaleFactor = dpr.getDoubleProperty("scaleFactor");
            int globalIndex = dpr.getIntProperty("globalIndex");
            const SerializationNode& a0Params = dpr.getChildNode("a0Params");
            int n = a0Params.getChildren().size();
            std::vector<double> a0(n);
            for (int j = 0; j < n; j++) {
                a0[j] = a0Params.getChildren()[j].getIntProperty("param");
            }
            const SerializationNode& a1Params = dpr.getChildNode("a1Params");
            n = a1Params.getChildren().size();
            std::vector<double> a1(n);
            for (int j = 0; j < n; j++) {
                a1[j] = a1Params.getChildren()[j].getIntProperty("param");
            }
            const SerializationNode& a2Params = dpr.getChildNode("a2Params");
            n = a2Params.getChildren().size();
            std::vector<double> a2(n);
            for (int j = 0; j < n; j++) {
                a2[j] = a2Params.getChildren()[j].getIntProperty("param");
            }
            const SerializationNode& a3Params = dpr.getChildNode("a3Params");
            n = a3Params.getChildren().size();
            std::vector<double> a3(n);
            for (int j = 0; j < n; j++) {
                a3[j] = a3Params.getChildren()[j].getIntProperty("param");
            }
            force->addDistProfileRestraint(atom1, atom2, rMin, rMax, nBins, a0, a1, a2, a3, scaleFactor);
        }

        // deserialize torsion profile restraints
        const SerializationNode& torsProfRestraints = node.getChildNode("TorsProfRestraints");
        for (int i = 0; i < (int) torsProfRestraints.getChildren().size(); i++) {
            const SerializationNode& tpr = torsProfRestraints.getChildren()[i];
            int atom1 = tpr.getIntProperty("atom1");
            int atom2 = tpr.getIntProperty("atom2");
            int atom3 = tpr.getIntProperty("atom3");
            int atom4 = tpr.getIntProperty("atom4");
            int atom5 = tpr.getIntProperty("atom5");
            int atom6 = tpr.getIntProperty("atom6");
            int atom7 = tpr.getIntProperty("atom7");
            int atom8 = tpr.getIntProperty("atom8");
            int nBins = tpr.getIntProperty("nBins");
            float scaleFactor = tpr.getDoubleProperty("scaleFactor");
            int globalIndex = tpr.getIntProperty("globalIndex");
            const SerializationNode& a0Params = tpr.getChildNode("a0Params");
            int n = a0Params.getChildren().size();
            std::vector<double> a0(n);
            for (int j = 0; j < n; j++) {
                a0[j] = a0Params.getChildren()[j].getIntProperty("param");
            }
            const SerializationNode& a1Params = tpr.getChildNode("a1Params");
            n = a1Params.getChildren().size();
            std::vector<double> a1(n);
            for (int j = 0; j < n; j++) {
                a1[j] = a1Params.getChildren()[j].getIntProperty("param");
            }
            const SerializationNode& a2Params = tpr.getChildNode("a2Params");
            n = a2Params.getChildren().size();
            std::vector<double> a2(n);
            for (int j = 0; j < n; j++) {
                a2[j] = a2Params.getChildren()[j].getIntProperty("param");
            }
            const SerializationNode& a3Params = tpr.getChildNode("a3Params");
            n = a3Params.getChildren().size();
            std::vector<double> a3(n);
            for (int j = 0; j < n; j++) {
                a3[j] = a3Params.getChildren()[j].getIntProperty("param");
            }
            const SerializationNode& a4Params = tpr.getChildNode("a4Params");
            n = a4Params.getChildren().size();
            std::vector<double> a4(n);
            for (int j = 0; j < n; j++) {
                a4[j] = a4Params.getChildren()[j].getIntProperty("param");
            }
            const SerializationNode& a5Params = tpr.getChildNode("a5Params");
            n = a5Params.getChildren().size();
            std::vector<double> a5(n);
            for (int j = 0; j < n; j++) {
                a5[j] = a5Params.getChildren()[j].getIntProperty("param");
            }
            const SerializationNode& a6Params = tpr.getChildNode("a6Params");
            n = a6Params.getChildren().size();
            std::vector<double> a6(n);
            for (int j = 0; j < n; j++) {
                a6[j] = a6Params.getChildren()[j].getIntProperty("param");
            }
            const SerializationNode& a7Params = tpr.getChildNode("a7Params");
            n = a7Params.getChildren().size();
            std::vector<double> a7(n);
            for (int j = 0; j < n; j++) {
                a7[j] = a7Params.getChildren()[j].getIntProperty("param");
            }
            const SerializationNode& a8Params = tpr.getChildNode("a8Params");
            n = a8Params.getChildren().size();
            std::vector<double> a8(n);
            for (int j = 0; j < n; j++) {
                a8[j] = a8Params.getChildren()[j].getIntProperty("param");
            }
            const SerializationNode& a9Params = tpr.getChildNode("a9Params");
            n = a9Params.getChildren().size();
            std::vector<double> a9(n);
            for (int j = 0; j < n; j++) {
                a9[j] = a9Params.getChildren()[j].getIntProperty("param");
            }
            const SerializationNode& a10Params = tpr.getChildNode("a10Params");
            n = a10Params.getChildren().size();
            std::vector<double> a10(n);
            for (int j = 0; j < n; j++) {
                a10[j] = a10Params.getChildren()[j].getIntProperty("param");
            }
            const SerializationNode& a11Params = tpr.getChildNode("a11Params");
            n = a11Params.getChildren().size();
            std::vector<double> a11(n);
            for (int j = 0; j < n; j++) {
                a11[j] = a11Params.getChildren()[j].getIntProperty("param");
            }
            const SerializationNode& a12Params = tpr.getChildNode("a12Params");
            n = a12Params.getChildren().size();
            std::vector<double> a12(n);
            for (int j = 0; j < n; j++) {
                a12[j] = a12Params.getChildren()[j].getIntProperty("param");
            }
            const SerializationNode& a13Params = tpr.getChildNode("a13Params");
            n = a13Params.getChildren().size();
            std::vector<double> a13(n);
            for (int j = 0; j < n; j++) {
                a13[j] = a13Params.getChildren()[j].getIntProperty("param");
            }
            const SerializationNode& a14Params = tpr.getChildNode("a14Params");
            n = a14Params.getChildren().size();
            std::vector<double> a14(n);
            for (int j = 0; j < n; j++) {
                a14[j] = a14Params.getChildren()[j].getIntProperty("param");
            }
            const SerializationNode& a15Params = tpr.getChildNode("a15Params");
            n = a15Params.getChildren().size();
            std::vector<double> a15(n);
            for (int j = 0; j < n; j++) {
                a15[j] = a15Params.getChildren()[j].getIntProperty("param");
            }
            force->addTorsProfileRestraint(atom1, atom2, atom3, atom4, atom5, atom6, atom7, atom8,
                                           nBins, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15,
                                           scaleFactor);
        }

        // deserialize GMM restraints
        const SerializationNode& gmmRestraints = node.getChildNode("GMMRestraints");
        for (const auto& r : gmmRestraints.getChildren()){
            int nPairs = r.getIntProperty("nPairs");
            int nComponents = r.getIntProperty("nComponents");
            int globalIndex = r.getIntProperty("globalIndex");
            float scale = r.getDoubleProperty("scale");

            std::vector<int> atomIndices;
            for(const auto& x : r.getChildNode("atomIndices").getChildren()) {
                atomIndices.push_back(x.getIntProperty("index"));
            }

            std::vector<double> weights;
            for(const auto& x : r.getChildNode("weights").getChildren()) {
                weights.push_back(x.getDoubleProperty("weight"));
            }

            std::vector<double> means;
            for(const auto& x : r.getChildNode("means").getChildren()) {
                means.push_back(x.getDoubleProperty("mean"));
            }

            std::vector<double> precOnDiag;
            for(const auto& x : r.getChildNode("precisionOnDiagonals").getChildren()) {
                precOnDiag.push_back(x.getDoubleProperty("prec"));
            }

            std::vector<double> precOffDiag;
            for(const auto& x : r.getChildNode("precisionOffDiagonals").getChildren()) {
                precOffDiag.push_back(x.getDoubleProperty("prec"));
            }
            force->addGMMRestraint(nPairs, nComponents, scale, atomIndices, weights,
                                   means, precOnDiag, precOffDiag);
        }


        // deserialize groups
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

        // deserialize collections
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
