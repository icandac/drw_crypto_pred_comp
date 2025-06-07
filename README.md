# DRW Crypto Prediction Kaggle Competition  
> Code base for our submission of the DRW Kaggle Competition for crypto prediction. [`this`](http://kaggle.com/competitions/drw-crypto-market-prediction/team)
 is the link to the competition.

## setup & environment

Follow these steps to get started when you fork or clone the project:

### 1. install [`uv`](https://github.com/astral-sh/uv/)

use `uv` to manage your virtual environments:

```bash
pip install uv
```

### 2. create a virtual environment

create an environment using python 3.10 (or your preferred version):

```bash
uv venv --python 3.10
```

### 3. activate the environment

activate the virtual environment:

```bash
source .venv/bin/activate
```

### 4. install dependencies

install all the required dependencies from the `requirements.txt` file:

```bash
uv pip install -e .
```

### 5. install pre-commits

```bash
pre-commit install
```

### 6. make the git commits and pushed automated

```bash
chmod +x git-auto.sh
```

now you can just do ./git-auto.sh to push commits.

## Goal  

In this competition, the aim is to build a model capable of predicting short-term crypto future price movements using our production feature data alongside publicly available market volume statistics. 

## Description of our approach

We will change here to how we approach to the project.

## News from the competition

There will be some updates here about how we are doing
