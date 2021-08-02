#!/bin/sh

source activate reaper
conda env list

conda env update -f env-reaper.yml
