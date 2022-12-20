#!/usr/bin/env bash
#
# Created on Thurs Oct 10 2019
#
# Authored by Joanna Leng who works at the University of Leeds who is funded by
# EPSRC as a Research Software Engineering Fellow (EP/R025819/1) for the PERPL 
# project.
#
# ---
# Copyright 2019 Peckham Lab
# 
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
#

SCRIPT_DIR=`dirname "$0"`
echo $0
echo $1
echo $SCRIPT_DIR
SCRIPT_FILE1="$SCRIPT_DIR/relative_positions.py"
echo $SCRIPT_FILE1
DATA_DIR="../data-perpl"
DATA_FILE="Nup107_SNAP_3D_GRROUPED_10nmZprec.txt"
DATA="$DATA_DIR/$DATA_FILE"
echo $DATA_DIR
echo $DATA_FILE
echo $DATA

echo "1)"
python3 "$SCRIPT_FILE1" blah
MY_CODE=$?
echo "Exit code ${MY_CODE}"

echo "2)"
python3 "$SCRIPT_FILE1" -h
MY_CODE=$?
echo "Exit code ${MY_CODE}"

echo "3)"
python3 "$SCRIPT_FILE1" -f
MY_CODE=$?
echo "Exit code ${MY_CODE}"

echo "4)"
python3 "$SCRIPT_FILE1" -i "$DATA" -f 100
MY_CODE=$?
echo "Exit code ${MY_CODE}"


SCRIPT_FILE2="$SCRIPT_DIR/rot_2d_symm_fit.py"
echo $SCRIPT_FILE2

echo "1)"
python3 "$SCRIPT_FILE2" blah
MY_CODE=$?
echo "Exit code ${MY_CODE}"

echo "2)"
python3 "$SCRIPT_FILE2" -h
MY_CODE=$?
echo "Exit code ${MY_CODE}"

echo "3)"
python3 "$SCRIPT_FILE2" -f 
MY_CODE=$?
echo "Exit code ${MY_CODE}"
