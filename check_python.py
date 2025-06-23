#!/usr/bin/env python3
import sys
import platform

print(f"Python version: {sys.version}")
print(f"Python version info: {sys.version_info}")
print(f"Platform: {platform.platform()}")
print(f"Python executable: {sys.executable}")

# Check if version is compatible
if sys.version_info >= (3, 10) and sys.version_info < (3, 12):
    print("✅ Python version is compatible with TensorFlow")
else:
    print("❌ Python version may have compatibility issues with TensorFlow")
    
# Check for distutils
try:
    import distutils
    print("✅ distutils is available")
except ImportError:
    print("❌ distutils is not available")
    try:
        import setuptools
        print("✅ setuptools is available as fallback")
    except ImportError:
        print("❌ Neither distutils nor setuptools available") 