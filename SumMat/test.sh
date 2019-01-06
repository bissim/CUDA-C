#!/bin/bash
if [ $# -lt 6 ]; then
    echo "Only $# parameters specified!"
    echo "Use of script: $0 <sourceFileName> <binaryFileName> <numberOfValues> <iterations> <outputFileName> <printFlag>"
    exit
fi

SRC=$1
BIN=$2

echo "Testing $SRC..."
echo

echo "Compiling $SRC in $BIN..."

nvcc -arch sm_20 $SRC -o $BIN
echo "$BIN compiled!"
echo

N=$3
ITERATIONS=$4
OUTPUT_FILE=$5
PRINT=$6

rm $OUTPUT_FILE
echo "$NUM_THREADS THREADS"
while [ $ITERATIONS -gt 0 ]; do
    echo "$N numbers"
    ./$BIN $N $NUM_THREADS $PRINT >> $OUTPUT_FILE
    let N=$N*2
    let ITERATIONS=$ITERATIONS-1
    echo "-----" >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE
done

echo "Output saved into $OUTPUT_FILE"

echo
echo "End of tests for $BIN"
