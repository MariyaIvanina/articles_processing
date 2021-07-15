@echo off

call activate reaper
call conda env list

call conda env update -f env-reaper.yml
