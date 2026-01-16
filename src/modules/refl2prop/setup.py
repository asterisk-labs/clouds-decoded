from setuptools import setup, find_packages

setup(
    name="refl2prop",
    version="0.1.0",
    description="Inversion model for radiative transfer: TOA Reflectance -> Cloud Properties",
    author="Alistair Francis",
    author_email="ali@asterisk.coop",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch>=1.10.0",
        "xarray",
        "netCDF4"
    ],
    entry_points={
        'console_scripts': [
            'refl2prop-train=refl2prop.train:main',
        ],
    },
    python_requires=">=3.8",
)