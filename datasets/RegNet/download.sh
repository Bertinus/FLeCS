#!/bin/bash

wget \
    -c \
    --secure-protocol=auto \
    --no-check-certificate \
    http://regnetworkweb.org/download/RegulatoryDirections.zip

unzip RegulatoryDirections.zip
rm RegulatoryDirections.zip
rm -rf __MACOSX
chmod 644 *.txt
