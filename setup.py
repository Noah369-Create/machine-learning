from setuptools import find_packages, setup
from typing import List


def get_requirements(file_path:str)->List[str]:
    '''
    This function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements




setup(
    name='machine-learning',
    version='0.0.1',
    author='Akalu',
    author_email='ake.abrish@gmail.com',
    description='This project is to predict the price of a used car based on its features.',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
    
)