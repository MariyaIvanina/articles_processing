@echo off

call conda env list
call conda create --name reaper python=3.8.10
call conda env list
