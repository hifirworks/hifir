#!/bin/bash

# script to parse a test script's first 50 lines into space/nl separated
# array of elements and extract out the dependencies
# Author(s):
#   Qiao,

set -e

lines=`head -n 50 ./${1}.cpp`
flag=0  # flag to indicate
deps=""
for l in $lines; do
    if [ "$l" = "#include" ]; then
        flag=1
    elif [ $flag -eq 1 ]; then
        if [[ $l =~ ^[\<\"]* ]]; then
             # remove first and last characters
             dep=`echo $l | sed 's/.//;s/.$//'`
             [[ $dep == psmilu_* ]] && deps="$deps $dep"
        fi
        flag=0
    elif [[ $l =~ ^#include[\<\"]* ]]; then
        # case where there is not space between include and < or "
        dep=`echo $l | sed 's/[<">]//g' | sed 's/#include//g'`
        [[ $dep == psmilu_* ]] && deps="$deps $dep"
    fi
done
echo $deps
