from setuptools import setup, find_packages
from meld import __version__


setup(
    name="Meld",
    version=__version__,
    author="Justin L. MacCallum",
    author_email="justin.maccallum@ucalgary.ca",
    packages=find_packages(),
    # meldplugin must be generated with SWIG
    install_requires=["mdtraj", "gemmi", "mrcfile", "matplotlib", "progressbar2", "meldplugin"],
    package_data={
        "meld.system.builders.amber": ["maps/*.txt"],
        "meld.test.test_functional.test_openmm_runner": [
            "system.top",
            "system.mdcrd",
        ],
    },
    scripts=[
        "scripts/analyze_energy",
        "scripts/analyze_remd",
        "scripts/extract_trajectory",
        "scripts/launch_remd",
        "scripts/process_fragments",
        "scripts/prepare_restart",
        "scripts/density_rank",
        "scripts/process_density_map",
        "scripts/cryofold2_setup",
    ],
    url="http://meldmd.org",
    license="LICENSE.txt",
    description="Moldeling Employing Limited Data",
    long_description=open("README.md").read(),
)
