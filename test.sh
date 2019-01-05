#!/bin/bash
if [ "$1" = "" ] || [ "$2" = "" ] || [ "$3" = "" ] || [ "$4" = "" ] || [ "$5" = "" ] || [ "$6" = "" ]; then
    echo "Use of script: $0 <sourceFileName> <binaryFileName> <numThreads> <numberOfValues> <iterations> <outputFileName> <printFlag>"
    exit
fi

SOURCE=$1
BINARY=$2

echo "Testing $SOURCE..."
echo

echo "Compiling $SOURCE in $BINARY..."

nvcc -arch sm_20 $SOURCE -o $BINARY
echo "$BINARY compiled!"
echo

NUM_THREADS=$3
N=$4
ITERATIONS=$5
OUTPUT_FILE=$6
PRINT=$7

rm $OUTPUT_FILE
echo "$NUM_THREADS THREADS"
while [ $ITERATIONS -gt 0 ]; do
    echo "$N numbers"
    ./$BINARY $N $NUM_THREADS $PRINT >> $OUTPUT_FILE
    let N=$N*2
    let ITERATIONS=$ITERATIONS-1
    echo "-----" >> $OUTPUT_FILE
    echo "" >> $OUTPUT_FILE
done

echo "Output saved into $OUTPUT_FILE"

echo
echo "End of tests for $BINARY"
