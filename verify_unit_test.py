#!/usr/bin/env python3
"""
Verify that the unit test we added works correctly.
"""

import sys
import os

# Add the vertexai lib to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'libs', 'vertexai'))

# Import the test function
from tests.unit_tests.test_vectorstores import test_to_data_points_with_integer_metadata

def main():
    """Run the unit test function."""
    print("Running test_to_data_points_with_integer_metadata...")
    
    try:
        test_to_data_points_with_integer_metadata()
        print("✓ Unit test passed successfully!")
        return True
    except Exception as e:
        print(f"✗ Unit test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 SUCCESS: The unit test is working correctly!")
        sys.exit(0)
    else:
        print("\n❌ FAILURE: The unit test failed.")
        sys.exit(1)
