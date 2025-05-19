
#!/usr/bin/env python3
"""
Patch keras-rl2 to work with newer versions of tensorflow/keras
"""
import sys
import os
import tensorflow as tf

# Create and apply the monkeypatch for callbacks.py
def patch_keras_rl():
    """Patches keras-rl2 to work with newer versions of TensorFlow/Keras"""
    # Find the callbacks.py file to patch
    import rl
    rl_path = os.path.dirname(rl.__file__)
    callbacks_path = os.path.join(rl_path, 'callbacks.py')
    
    # Read the file
    with open(callbacks_path, 'r') as f:
        content = f.read()
    
    # Replace the problematic import
    patched_content = content.replace(
        'from tensorflow.keras import __version__ as KERAS_VERSION',
        'KERAS_VERSION = "{}"'.format(tf.__version__)
    )
    
    # Write to a temporary file
    patch_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'patched_rl')
    os.makedirs(patch_dir, exist_ok=True)
    patched_path = os.path.join(patch_dir, 'callbacks.py')
    
    with open(patched_path, 'w') as f:
        f.write(patched_content)
    
    # Create __init__.py to make the directory a package
    with open(os.path.join(patch_dir, '__init__.py'), 'w') as f:
        f.write('')
    
    # Add the patch directory to the path
    sys.path.insert(0, os.path.dirname(patch_dir))
    
    # Replace the module in sys.modules
    import patched_rl.callbacks
    sys.modules['rl.callbacks'] = patched_rl.callbacks
    
    print("Applied keras-rl2 compatibility patch")

# Apply the patch
if __name__ == "__main__":
    patch_keras_rl()
