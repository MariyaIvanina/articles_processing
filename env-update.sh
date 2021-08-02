#!/bin/sh

conda activate usaid
conda env list

conda env update -f env-usaid.yml
