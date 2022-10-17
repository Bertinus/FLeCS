# Informations


## KEGG pathway

As an example, we use the Calcium signaling pathway in Homo sapiens. 

File `datasets/KEGG/hsa04020.xml` downloaded from https://www.kegg.jp/kegg-bin/download?entry=hsa04020&format=kgml

Please visit https://www.genome.jp/pathway/hsa04020 for more information about this pathway.


## ATAC

### ATAC-seq dataset for Homo sapiens K562

Downloaded from https://www.encodeproject.org/experiments/ENCSR483RKN/ on 2022/09/09

### Gene information

Gene information downloaded from 
https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_41/gencode.v41.annotation.gff3.gz

(see https://www.gencodegenes.org/pages/data_access.html for more info)

Select lines which correspond to genes only using the following command:
`cat gencode.v41.annotation.gff3 | awk '$3 == "gene"'  > genes.gff3
`

## RealNet

Download the following zip file: http://www2.unil.ch/cbg/regulatorycircuits/Network_compendium.zip


## RegNet

File `new_kegg.human.reg.direction.txt` downloaded from http://regnetworkweb.org/download/RegulatoryDirections.zip


## REgulonDB

File `network_tf_gene.txt` downloaded from http://regulondb.ccg.unam.mx/menu/download/datasets/files/network_tf_gene.txt

File `network_tf_tf.txt` downloaded from http://regulondb.ccg.unam.mx/menu/download/datasets/files/network_tf_tf.txt
