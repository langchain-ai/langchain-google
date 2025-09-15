#!/usr/bin/env python3
"""
Debug script to understand the structure of NumericRestriction objects.
"""

import sys
import os

# Add the vertexai lib to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'libs', 'vertexai'))

from langchain_google_vertexai.vectorstores._utils import to_data_points

def debug_numeric_restrictions():
    """Debug the structure of NumericRestriction objects."""
    print("Debugging NumericRestriction structure...")
    
    # Test data with both integer and float values
    ids = ["test_id"]
    embeddings = [[0.1, 0.2, 0.3]]
    metadatas = [
        {
            "integer_field": 42,
            "float_field": 3.14,
        }
    ]
    
    # Call the function
    result = to_data_points(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    
    datapoint = result[0]
    
    print(f"Number of numeric restrictions: {len(datapoint.numeric_restricts)}")
    
    for i, restriction in enumerate(datapoint.numeric_restricts):
        print(f"\nRestriction {i}:")
        print(f"  Namespace: {restriction.namespace}")
        print(f"  Type: {type(restriction)}")
        print(f"  Dir: {[attr for attr in dir(restriction) if not attr.startswith('_')]}")
        
        # Check all possible value attributes
        for attr in ['value_int', 'value_float', 'value_double']:
            if hasattr(restriction, attr):
                value = getattr(restriction, attr)
                print(f"  {attr}: {value} (type: {type(value)})")
            else:
                print(f"  {attr}: NOT PRESENT")

if __name__ == "__main__":
    debug_numeric_restrictions()
