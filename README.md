# DRW Crypto Prediction Kaggle Competition  
Code base for our submission of the DRW Kaggle Competition for crypto prediction. [`This`](http://kaggle.com/competitions/drw-crypto-market-prediction/team) is the link to the competition.

## setup & environment

Follow these steps to get started when you fork or clone the project:

### 1. install [`uv`](https://github.com/astral-sh/uv/)

Use `uv` to manage your virtual environments:

```bash
pip install uv
```

### 2. create a virtual environment

Create an environment using python 3.10 (or your preferred version):

```bash
uv venv --python 3.10
```

### 3. activate the environment

Activate the virtual environment:

```bash
source .venv/bin/activate
```

### 4. install dependencies

Install all the required dependencies from the `requirements.txt` file:

```bash
uv pip install -e .
```

Sometimes we can forget updating the requirements.txt. If there are any missing package errors easily do:

```bash
uv pip install package_name
```

### 5. install pre-commits

```bash
pre-commit install
```

### 6. make the git commits and pushed automated

```bash
chmod +x git-auto.sh
```

Now you can just do ./git-auto.sh to push commits.

### 7. Download the data

The data is [`here`](https://www.kaggle.com/competitions/drw-crypto-market-prediction/data). You should download data and place it to a folder called 'data' in the parent folder in order to have no errors raising due to the data paths.

### 8. Run the code

The new methods of predictions should be placed under src/scripts whereas the utility functions and classes should be under src/utils. Somewhere in your method, there should be a runner function. Then, please put the name of that function into src/main.py as below, 

```bash
if __name__ == "__main__":
    lgbm_predictor.lgbm_runner()
```

then run 

```bash
python3 src/main.py
```

## Goal  

In this competition, the aim is to build a model capable of predicting short-term crypto future price movements using our production feature data alongside publicly available market volume statistics. 

## Description of our approach

We will change here to how we approach to the project.

## News from the competition

Until the last weekend the best Pearson Coefficient on the leaderboard was 0.27 which seems like weak. However, this makes sense to us since the test data is huge, around 500k rows, and although there are many columns to be used as features for the prediction, when the stochastic nature of the financial markets is taken into consideration, to have a prediction better than predicting flipping coins is almost nonsense. However, some new player came and crashed the coefficient with 0.4. Let's see how this evolves...