#!/bin/sh
python3 test_knn_generate.py     > ../test/knn.cpp
python3 test_votess_generate.py  > ../test/votess.cpp
