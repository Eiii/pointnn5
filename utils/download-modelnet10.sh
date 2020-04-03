#!/usr/bin/env bash

TMPFILE=$(mktemp)
curl -o $TMPFILE http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip 
unzip $TMPFILE -d data/
rm $TMPFILE
