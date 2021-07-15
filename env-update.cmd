@echo off

call activate usaid
call conda env list

call conda env update -f env-usaid.yml
