import pandas as pd
import gzip
import shutil
import os

# Your file
input_file = 'data_new_2025.csv'
output_file = 'data_new_2025.csv.gz'

# Check original size
orig_size = os.path.getsize(input_file) / (1024 * 1024)
print(f"Original CSV size: {orig_size:.2f} MB")

# Option A: Compress with gzip (keep as CSV but zipped)
with open(input_file, 'rb') as f_in:
    with gzip.open(output_file, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

# Check compressed size
new_size = os.path.getsize(output_file) / (1024 * 1024)
print(f"Compressed size: {new_size:.2f} MB")
print(f"Saved: {orig_size - new_size:.2f} MB ({(1 - new_size/orig_size)*100:.1f}%)")
print(f"\nCompressed file: {output_file}")