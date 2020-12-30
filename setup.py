"""
setup.py for the Sagitta PyPI package
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name="Sagitta",
    version="0.1",
    author="Aidan McBride, Ryan Lingg, Marina Kounkel, Kevin Covey, Brian Hutchinson",
    author_email="mcbrida5@wwu.edu, linggr@wwu.edu, marina.kounkel@wwu.edu",
    description="A neural network based pipeline to identify pre-main sequence stars and estimate their ages.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hutchresearch/Sagitta",
    packages=['sagitta', 'sagitta.tests'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_data={
        'sagitta' : ['model_state_dicts/av_model.pt','model_state_dicts/pms_model.pt','model_state_dicts/age_model.pt'],
        'data_tools' : ['data_tools.py'],
        "model_code" : ['model_code.py'],
        'sagitta.tests' : []
    },
    install_requires=['numpy','torch','astropy','astroquery','pandas','galpy'],
    entry_points={'console_scripts': ['sagitta=sagitta.sagitta:main']},
    python_requires='>=3.6',
)