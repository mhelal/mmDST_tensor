#!/bin/bash
# download_data.sh - Download and prepare all datasets

cd data

# BAliBASE
echo "Downloading BAliBASE..."
wget -c http://www.lbgi.fr/balibase/Balibase_R1-5.tar.gz
tar -xzf Balibase_R1-5.tar.gz
wget -c http://www.lbgi.fr/balibase/Balibase_R6-8.tar.gz
tar -xzf Balibase_R6-8.tar.gz
wget -c http://www.lbgi.fr/balibase/Balibase_R9.tar.gz
tar -xzf Balibase_R9.tar.gz

# JASPAR
echo "Downloading JASPAR..."
python -c "
from pyjaspar import jaspardb
import os
os.makedirs('jaspar', exist_ok=True)
jdb = jaspardb(release='JASPAR2024')
motifs = jdb.fetch_motifs(collection='CORE', tax_group=['vertebrates'])
for motif in motifs:
    motif.to_pfm(f'jaspar/{motif.name}.pfm')
print(f'Downloaded {len(motifs)} motifs')
"

# Upstream sequences (if available)
echo "Preparing upstream sequences..."
if [ -f "../code/data/motif_databases/upstream1000.fa" ]; then
    cp ../code/data/motif_databases/upstream1000.fa .
else
    echo "upstream1000.fa not found, creating synthetic..."
    python -c "
import random
random.seed(42)
seqs = [''.join(random.choice('ACGT') for _ in range(1000)) for _ in range(100)]
with open('upstream1000.fa', 'w') as f:
    for i, seq in enumerate(seqs):
        f.write(f'>SEQ_{i}\n{seq}\n')
print('Created synthetic upstream1000.fa')
"
fi

echo "Data download complete!"
