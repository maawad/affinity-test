#!/bin/bash
for i in $(seq 2 2 42)
do
    ./build/affinity_test $i
done