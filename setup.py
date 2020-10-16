# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['fclearn']

package_data = \
{'': ['*']}

install_requires = \
['flake8-import-order>=0.18.1,<0.19.0',
 'pandas==0.23',
 'seaborn>=0.11.0,<0.12.0',
 'sklearn>=0.0,<0.1',
 'statsmodels>=0.12.0,<0.13.0']

setup_kwargs = {
    'name': 'fclearn',
    'version': '0.1.0',
    'description': 'Code written for Master Thesis',
    'long_description': None,
    'author': 'Lars Hanegraaf',
    'author_email': 'larshanegraaf@live.nl',
    'maintainer': None,
    'maintainer_email': None,
    'url': 'https://github.com:LarsHanegraaf/fclearn.git',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.6,<4.0',
}


setup(**setup_kwargs)
