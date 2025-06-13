#!/bin/bash
clear


echo "Extracting obj from IFC files:"
python3 ./m03_Data_PreProcessing/01_extract_ifc.py

echo "Generating SDF from obj files:"
python3 ./m03_Data_PreProcessing/02_extract_sdf.py
