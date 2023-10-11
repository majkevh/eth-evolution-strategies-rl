# Evolution Strategies in Reinforcement Learning

## Project structure
- ``presentation.pdf`` - The Project Keynote presentation
- ``evolution`` - The Python module and files.
- ``tests`` - The Python module for testing the code.
- ``evolution/simulation`` - Folders where the raw data will be generated.
- ``evolution/base`` - Base class containing all the model helper and initialization functions.
- ``evolution/em`` - Inheritance class containing the algorithm structure and the export of the data.
- ``evolution/parser`` - Contains a parser to run a simulation.
- ``evolution/plot`` - Data import and plotting.
- ``evolution/run.sh`` - Bash script to use to launch benchmark simulations.
- ``evolution/reinforce.sh`` - Bash script to use to launch reinforcement learning simulations.

## Compilation

Install all required packages running

```python
pip3 install -r requirements.txt --user
```

In order to run reinforcement learning simulations, you need to install additionally MuJoCo following [these instructions](https://github.com/openai/mujoco-py).

## Tests
To run the tests, run the following command (from the repository root):

```python
cd tests
python3 test_all.py
```

## Troubleshooting 
If the following error appear while running the tests:

```python
error: command '/usr/local/bin/gcc-9' failed with exit status 1
```
try to install an older [python version (< 3.9.5)](https://www.python.org/downloads/) and delete the current one. On macOS you can check where isyour Python installation running:

```python
which python3 
```

 Additionally you may need to remove your current version of gcc and reinstall it again. On macOS you can check where is your current installation of gcc running:

```python
which gcc
```
For installing again gcc, run:
```python
brew install gcc
```

## Usage
 In order to run optimization problems with benchmark functions or reinforcement learning problems, run (from the repository root):

```python
cd py
bash run.sh #bash reinforce.sh
```
A new folder called ``data`` will be generated, containing all the raw data from the simulations. Additionally a plot with the total error (or cumulative reward respectively) will be generated in the ``py/simulation/plots`` folder. 



