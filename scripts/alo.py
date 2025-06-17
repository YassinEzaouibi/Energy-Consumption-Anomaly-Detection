import platform

py_version = platform.python_version()
print(f"Python version: {py_version}")

tf_compatible = False
if py_version.startswith("3."):
    minor_version = int(py_version.split(".")[1])
    if 8 <= minor_version <= 11:
        tf_compatible = True

print(f"Python version compatible with TensorFlow: {tf_compatible}")
