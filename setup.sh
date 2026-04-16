#!/bin/bash 
git clone https://github.com/FREVA-CLINT/FieldSpaceNN.git

cd FieldSpaceNN

git checkout FST_initial_paper

python -m venv .venv

source .venv/bin/activate

pip install .