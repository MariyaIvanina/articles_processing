#!/bin/sh

conda activate reaper
conda env list

conda env update -f env-reaper.yml
