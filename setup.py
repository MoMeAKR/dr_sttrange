from setuptools import setup, find_packages

setup(
    name='mome_dt',  
    version='0.1.0',    
    author='mOmE', 
    description='Utilities to build decision trees from dict',
    
    packages=find_packages(),  
    install_requires=[        
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)
