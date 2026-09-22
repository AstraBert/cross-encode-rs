#!/bin/bash

turns=$1
name=$2

mkdir -p results/${name}/

# warmup
echo "====================== WARMUP ======================"
echo ""
go run main.go 100 $ENDPOINT
echo ""
echo "===================================================="
echo ""

for i in $(seq 0 $turns)
do
    echo "====================== RUN ${i} ======================"
    echo ""
    for num_requests in 1000 10000
    do
        go run main.go $num_requests $ENDPOINT >> results/${name}/${i}.txt
    done
    echo ""
    echo "======================================================"
    echo ""
done
