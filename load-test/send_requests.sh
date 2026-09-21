#!/bin/bash

# warmup
go run main.go 100 $ENDPOINT

for num_requests in 1000 10000
do
    go run main.go $num_requests $ENDPOINT
done
