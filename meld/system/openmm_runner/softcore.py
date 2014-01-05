#!/usr/bin/env python
# encoding: utf-8

from simtk import openmm as mm
from simtk.openmm import app
from simtk.unit import *
from sys import stdout
from xml.etree import ElementTree as ET


def _extract_force(root, force_name):
    forces = root.find('Forces')
    force = forces.findall('./Force[@type="{}"]'.format(force_name))
    if force:
        assert len(force) == 1
        force = force[0]
        forces.remove(force)
        return force
    else:
        return None


def _get_sc_nb_force(charges, sigmas, epsilons, sigma_sc, exceptions):
    force = mm.CustomNonbondedForce(
        'qq_lambda * 138.935485 * q1 * q2 / r + lj_lambda * 4 * eps * (sig^12 / r_eff^12 - sig^6 / r_eff^6);'
        'r_eff = ( 0.5 * (1 - sc_lambda) * sig_sc^6 + r^6)^(1/6);'
        'sig = 0.5 * (sigma1 + sigma2);'
        'eps = sqrt(epsilon1 * epsilon2);'
        'sig_sc = 0.5 * (sigma_sc1 + sigma_sc2)')
    force.addGlobalParameter('sc_lambda', 1.)
    force.addGlobalParameter('qq_lambda', 1.)
    force.addGlobalParameter('lj_lambda', 1.)
    force.addPerParticleParameter('q')
    force.addPerParticleParameter('sigma')
    force.addPerParticleParameter('epsilon')
    force.addPerParticleParameter('sigma_sc')

    for q, sig, eps, sig_sc in zip(charges, sigmas, epsilons, sigma_sc):
        force.addParticle([q, sig, eps, sig_sc])

    for i, j, q, s, e in exceptions:
        force.addExclusion(i, j)

    return force


def _get_sc_nb_exception_force(exceptions, sigma_sc):
    force = mm.CustomBondForce(
        'qq_lambda * 138.935485 * qq / r + lj_lambda * 4 * eps * (sig^12 / r_eff^12 - sig^6 / r_eff^6);'
        'r_eff = (0.5 * (1 - sc_lambda) * sigma_sc^6 + r^6) ^ (1/6);')

    force.addGlobalParameter('sc_lambda', 1.)
    force.addGlobalParameter('qq_lambda', 1.)
    force.addGlobalParameter('lj_lambda', 1.)
    force.addPerBondParameter('qq')
    force.addPerBondParameter('sig')
    force.addPerBondParameter('eps')
    force.addPerBondParameter('sigma_sc')

    for i, j, q, s, e in exceptions:
        sig_sc = 0.5 * (sigma_sc[i] + sigma_sc[j])
        force.addBond(i, j, [q, s, e, sig_sc])
    return force


def _get_sc_gb_force(charges, radii, scale, solventDielectric=78.5, soluteDielectric=1, SA=None, cutoff=None):
    force = mm.CustomGBForce()

    force.addPerParticleParameter('q')
    force.addPerParticleParameter('radius')
    force.addPerParticleParameter('scale')
    force.addGlobalParameter('qq_lambda', 1.)
    force.addGlobalParameter('solventDielectric', solventDielectric)
    force.addGlobalParameter('soluteDielectric', soluteDielectric)
    force.addGlobalParameter('offset', 0.009)
    force.addComputedValue('I', 'qq_lambda * step(r+sr2-or1)*0.5*(1/L-1/U+0.25*(r-sr2^2/r)*(1/(U^2)-1/(L^2))+0.5*log(L/U)/r);'
                                'U=r+sr2;'
                                'L=max(or1, D);'
                                'D=abs(r-sr2);'
                                'sr2 = scale2*or2;'
                                'or1 = radius1-offset; or2 = radius2-offset;', mm.CustomGBForce.ParticlePairNoExclusions)

    force.addComputedValue('B', '1/(1/or-tanh(psi-0.8*psi^2+4.85*psi^3)/radius);'
                                'psi=I*or; or=radius-offset', mm.CustomGBForce.SingleParticle)

    force.addEnergyTerm('qq_lambda * -0.5*138.935485*(1/soluteDielectric-1/solventDielectric)*q^2/B', mm.CustomGBForce.SingleParticle)
    if SA == 'ACE':
        force.addEnergyTerm('qq_lambda * 28.3919551*(radius+0.14)^2*(radius/B)^6', mm.CustomGBForce.SingleParticle)
    elif SA is not None:
        raise ValueError('Unknown surface area method: ' + SA)
    if cutoff is None:
        force.addEnergyTerm('qq_lambda * -138.935485*(1/soluteDielectric-1/solventDielectric)*q1*q2/f;'
                            'f=sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)));', mm.CustomGBForce.ParticlePairNoExclusions)
    else:
        force.addEnergyTerm('qq_lambda * -138.935485*(1/soluteDielectric-1/solventDielectric)*q1*q2*(1/f-' + str(1 / cutoff) + ');'
                            'f=sqrt(r^2+B1*B2*exp(-r^2/(4*B1*B2)));', mm.CustomGBForce.ParticlePairNoExclusions)

    for c, r, s in zip(charges, radii, scale):
        force.addParticle([c, r, s])

    return force


def add_soft_core(system, sigma_min=0.151):
    system_xml = mm.XmlSerializer.serializeSystem(system)

    # grab and remove the non-bonded force
    root = ET.fromstring(system_xml)
    nb_string = _extract_force(root, 'NonbondedForce')
    gb_string = _extract_force(root, 'GBSAOBCForce')

    nb_force = mm.XmlSerializer.deserialize(ET.tostring(nb_string))

    n_particles = nb_force.getNumParticles()

    # extract the non-bonded parameters
    nb_params = [nb_force.getParticleParameters(i) for i in range(n_particles)]
    charges = [params[0] for params in nb_params]
    sigmas = [params[1] for params in nb_params]
    epsilons = [params[2] for params in nb_params]
    exceptions = [nb_force.getExceptionParameters(i) for i in range(nb_force.getNumExceptions())]

    sigma_sc = [sigma_min * nanometer if s == 0 else s for s in sigmas]

    sc_nb_force = _get_sc_nb_force(charges, sigmas, epsilons, sigma_sc, exceptions)
    sc_nb_exclusion_force = _get_sc_nb_exception_force(exceptions, sigma_sc)

    new_system = mm.XmlSerializer.deserializeSystem(ET.tostring(root))
    new_system.addForce(sc_nb_force)
    new_system.addForce(sc_nb_exclusion_force)

    # extract the gb_parameters
    if gb_string is not None:
        gb_force = mm.XmlSerializer.deserialize(ET.tostring(gb_string))
        gb_params = [gb_force.getParticleParameters(i) for i in range(n_particles)]
        gb_charge = [p[0] for p in gb_params]
        gb_radius = [p[1] for p in gb_params]
        gb_scale = [p[2] for p in gb_params]
        sc_gb_force = _get_sc_gb_force(gb_charge, gb_radius, gb_scale, SA='ACE')
        new_system.addForce(sc_gb_force)
    return new_system
