#!/bin/bash

CSV_DIR=$1
sed -i'' 's/i/j/g' $CSV_DIR/*spin_cov*.csv
sed -i'' 's/trj/tri/g' $CSV_DIR/*spin_cov*.csv
