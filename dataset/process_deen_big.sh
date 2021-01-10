#!/bin/env bash

# tokenize data
./tokenize_deen_big.py

# generate one file data_deen_big.sent out of TildeMODEL dataset folder
paste TildeMODEL/de.tok TildeMODEL/en.tok | sed -e 's/\t/ ||| /g' > data_deen_big.sent

echo -n "Sentences: "
cat TildeMODEL/de.tok | wc -l

echo -n "Tokens DE: "
cat TildeMODEL/de.tok | tr " " "\n" | wc -l

echo -n "Tokens EN: "
cat TildeMODEL/en.tok | tr " " "\n" | wc -l