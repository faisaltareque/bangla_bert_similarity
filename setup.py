from setuptools import setup, find_packages

VERSION = '0.0.1'
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='bangla_sentence_similarity',
    version=VERSION,
    url='https://github.com/faisaltareque/helloworld',
    author='Faisal Tareque',
    author_email='faisaltareque98@gmail.com',
    description='Bangla Sentence Similarity Score',
    long_description=long_description,
    long_description_content_type="text/markdown",
    # py_modules=['bangla_bert_similarity'],
    # package_dir={'':'src'},
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    # install_requires = [
    #     'normalizer @ git+https://github.com/csebuetnlp/normalizer@c3657b366d3289517d8756f55439cbab6bee5ee2',
    #     'scikit-learn',
    #     'transformers==4.18.0'
    # ],
    extras_require ={
        "dev":[
            "pytest >=3.7",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License:: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)