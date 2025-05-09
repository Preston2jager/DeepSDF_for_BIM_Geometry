#!/bin/bash
clear


echo "Extracting obj from IFC files:"
python3 ./m03_Data_PreProcessing/extract_ifc.py

echo "Generating SDF from obj files:"
python3 ./m03_Data_PreProcessing/extract_sdf.py
