import os
import gc
import psutil

# Memory optimization
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

def get_memory():
    gc.collect()
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

print(f"📊 Memory before TensorFlow: {get_memory():.1f}MB")

import tensorflow as tf

print(f"📊 Memory after TensorFlow import: {get_memory():.1f}MB")

# Disable eager execution
tf.config.run_functions_eagerly(False)

print(f"📊 Memory after optimization: {get_memory():.1f}MB")