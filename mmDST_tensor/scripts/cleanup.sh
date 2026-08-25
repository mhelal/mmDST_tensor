#!/bin/bash
# cleanup.sh - Remove adhoc files and organize

echo "Cleaning up adhoc files..."

# Move adhoc files to backup
mkdir -p ../backup_adhoc
mv ../code/*.py ../backup_adhoc/
mv ../code/*.csv ../backup_adhoc/
mv ../code/*.json ../backup_adhoc/
mv ../code/*.png ../backup_adhoc/

# Remove empty directories
find . -type d -empty -delete

echo "Cleanup complete!"
