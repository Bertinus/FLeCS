#!/bin/bash

# See https://www.gencodegenes.org/pages/data_access.html for more info!
wget -c https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_41/gencode.v41.annotation.gff3.gz
gunzip gencode.v41.annotation.gff3.gz
cat gencode.v41.annotation.gff3 | awk '$3 == "gene"'  > genes.gff3
rm gencode.v41.annotation.gff3
