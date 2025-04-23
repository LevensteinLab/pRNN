from setuptools import setup, find_packages


dependencies = [
    "pandas==1.5.3",
    "matplotlib==3.5.3",
    "scipy==1.12.0",
    "numpy==1.22.4",
    "scikit-learn==1.4.0",
    "pynapple==0.6.1",
    "zipp==3.17.0",
    "pyparsing==3.1.1",
    "importlib_metadata==7.0.1",
    "ruamel.yaml==0.18.6",
    #"gym==0.21.0 --no-binary :all:", TODO: Need to fix this
    "gymnasium==0.29.1",
    "ratinabox==1.7.1"
]

setup(
    author="Daniel Levenstein",
    author_email='daniel.levenstein@mila.quebec',
    python_requires='>=3.9',
    name='prnn',
    version='v0.1',
    packages=find_packages(),
    install_requires=dependencies,
    description="Python Library for Predictive RNNs Modeling Hippocampal Representation and Replay",
    license="MIT License",
    # Add more metadata like author, description, etc. as needed
)
