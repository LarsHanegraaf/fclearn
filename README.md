# fclearn
![Tests](https://github.com/LarsHanegraaf/fclearn/workflows/Tests/badge.svg?branch=master)
[![codecov](https://codecov.io/gh/LarsHanegraaf/fclearn/branch/master/graph/badge.svg?token=8F5LS5M58D)](undefined)

Code written for my Thesis. Forcasting in a scikit-learn manner.

Docs available at [ReadTheDocs](https://fclearn.readthedocs.io/en/latest/source/modules.html)
## Install
- Download the package
- Run setup.py


## Development

- Create a virtual environment using poetry
- Install nox and nox-poetry globally
- run `pythom -m nox -r` out of the virtual environment to run the tests

- Add flake8 as dev dependency
- Run `python -m nox -rs lint` to run lint session
- Set IDE with [following settings](https://py-vscode.readthedocs.io/en/latest/files/linting.html)

- Install [isort](https://pycqa.github.io/isort/docs/configuration/config_files/) and isort-flake8, and add configuration to pyproject.toml
- Edit vscode to [format on save](https://medium.com/@cereblanco/setup-black-and-isort-in-vscode-514804590bf9)
