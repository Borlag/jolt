
import pytest
import sys
import os

if __name__ == "__main__":
    if os.path.exists("SUCCESS"): os.remove("SUCCESS")
    if os.path.exists("FAILURE"): os.remove("FAILURE")
    
    ret = pytest.main(["tests/test_joint.py", "-q"])
    
    if ret == 0:
        with open("SUCCESS", "w") as f:
            f.write("PASSED")
    else:
        with open("FAILURE", "w") as f:
            f.write(f"FAILED: {ret}")
