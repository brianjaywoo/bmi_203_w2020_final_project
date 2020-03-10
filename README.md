# Build status

[![Build
Status](https://travis-ci.org/brianjaywoo/bmi_203_w2020_final_project.svg?branch=master)](https://travis-ci.org/brianjaywoo/bmi_203_w2020_final_project)

```
```
## Using this

To use the package, first make a new conda environment and activate it

```
conda create -n exampleenv python=3
source activate exampleenv
```

then run

```
conda install --yes --file requirements.txt
```

to install all the dependencies in `requirements.txt`. Then the package's
main function (located in `example/__main__.py`) can be run as follows

```
python -m scripts
```

## testing

Testing is as simple as running

```
python -m pytest
```

from the root directory of this project.
