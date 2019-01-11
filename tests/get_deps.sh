#!/bin/bash

line1=`head -n 1 ./${1}.cpp`
deps=""
for dep in $line1; do
    [[ $dep != //* ]] && deps="$deps $dep"
done
echo $deps