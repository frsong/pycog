#! /bin/bash
python setup.py install --record install_files.txt
cat install_files.txt | xargs rm -rf
