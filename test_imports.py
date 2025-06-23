#!/usr/bin/env python3
"""
Test script to verify all required packages can be imported correctly.
"""

def test_imports():
    """Test all required package imports"""
    
    print("Testing package imports...")
    
    try:
        import streamlit as st
        print("✓ Streamlit imported successfully")
    except ImportError as e:
        print(f"✗ Streamlit import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✓ Pandas {pd.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        import plotly
        print(f"✓ Plotly {plotly.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Plotly import failed: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    try:
        import sklearn
        print(f"✓ Scikit-learn {sklearn.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Scikit-learn import failed: {e}")
        return False
    
    try:
        import joblib
        print(f"✓ Joblib {joblib.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Joblib import failed: {e}")
        return False
    
    try:
        import librosa
        print(f"✓ Librosa {librosa.version.version} imported successfully")
    except ImportError as e:
        print(f"✗ Librosa import failed: {e}")
        return False
    except AttributeError:
        print("✓ Librosa imported successfully (version not accessible)")
        pass
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ TensorFlow import failed: {e}")
        return False
    
    print("\n✓ All packages imported successfully!")
    return True

if __name__ == "__main__":
    success = test_imports()
    if not success:
        exit(1)
    print("🎉 Import test completed successfully!") 