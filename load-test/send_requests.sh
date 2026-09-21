#!/bin/bash

for num_requests in 1000 10000 100000 1000000
do
    go run main.go $num_requests
done
