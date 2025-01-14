import sys
import dlltracer
with dlltracer.Trace(out=sys.stdout):
    import triton