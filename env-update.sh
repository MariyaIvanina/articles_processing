#!/bin/sh

source activate usaid
conda env list

conda env update -f env-usaid.yml
