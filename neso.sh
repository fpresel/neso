#!/bin/sh
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
#echo $DIR
cd "$DIR"
python Neso.py
