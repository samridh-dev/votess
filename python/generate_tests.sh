#!/bin/sh
python3 test_votess_generate.py  > ../test/votess.cpp
exit 0
python3 test_knn_generate.py     > ../test/knn.cpp
