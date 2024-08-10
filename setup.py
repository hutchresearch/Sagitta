"""
setup.py for the Sagitta PyPI package
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name="Sagitta",
    version="1.2.3",
    author="Aidan McBride, Ryan Lingg, Marina Kounkel, Kevin Covey, Brian Hutchinson",
    author_email="marina.kounkel@vanderbilt.edu, mcbrida5@wwu.edu, linggr@wwu.edu",
    description="A neural network based pipeline to identify pre-main sequence stars and estimate their ages.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hutchresearch/Sagitta",
    packages=['sagitta'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_data={
        'sagitta' : ['data_tools.py','model_code.py','model_state_dicts/av_model.pt','model_state_dicts/pms_model.pt','model_state_dicts/age_model.pt'],
        'sagitta.tests' : []
    },
    install_requires=['numpy','torch','astropy','astroquery','galpy'],
    entry_points={'console_scripts': ['sagitta=sagitta.sagitta:main']},
    python_requires='>=3.6',
)
