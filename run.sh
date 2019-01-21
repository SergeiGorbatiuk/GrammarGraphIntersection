#!/usr/bin/env bash
INPUT_CHOMSKY_FILE=$1
INPUT_GRAPH_FILE=$2
OUTPUT_FILE=$3

# call your programm here
touch $OUTPUT_FILE

python main.py $INPUT_CHOMSKY_FILE $INPUT_GRAPH_FILE $OUTPUT_FILE -s -type uint8