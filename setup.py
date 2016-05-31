import os
from setuptools import setup, find_packages

setup(
    name = "cluster_webapp",
    version = "0.3",
    author = "MJ Berends",
    author_email = "mj@dose.com",
    url = "",

    packages = find_packages('.'),
    package_dir = {'':'.'},
    data_files=[('.', ['README',]),],

    install_requires = [
        'regex',
        'pymongo',
        'mongoengine',
        'numpy',
        'pytables',
        'pandas',
        'termcolor',
        'tornado',
    ],

    package_data = {},
    include_package_data=True,

    keywords = "",
    description = "Web frontend and backend logic for clustering data hierarchically.",
    classifiers = [
        "Intended Audience :: Developers",
        'Programming Language :: Python',
    ]
)
