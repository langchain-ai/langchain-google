#!/usr/bin/env python3
"""
Simple test script to verify that integer metadata values are properly handled
in the VectorSearchVectorStore restrict filter fix.
"""

import sys
import os

# Add the vertexai lib to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'libs', 'vertexai'))

from langchain_google_vertexai.vectorstores._utils import to_data_points

def test_integer_metadata_handling():
    """Test that integer metadata values use value_int and float values use value_float."""
    print("Testing integer metadata handling...")
    
    # Test data with both integer and float values
    ids = ["test_id"]
    embeddings = [[0.1, 0.2, 0.3]]
    metadatas = [
        {
            "integer_field": 42,
            "float_field": 3.14,
            "another_integer": 100,
            "string_field": "test_string",
        }
    ]
    
    # Call the function
    result = to_data_points(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    
    # Verify basic structure
    assert len(result) == 1, f"Expected 1 result, got {len(result)}"
    datapoint = result[0]
    
    print(f"Datapoint ID: {datapoint.datapoint_id}")
    print(f"Number of numeric restrictions: {len(datapoint.numeric_restricts)}")
    print(f"Number of string restrictions: {len(datapoint.restricts)}")
    
    # Create lookup for numeric restrictions
    num_restrictions = {
        restriction.namespace: restriction
        for restriction in datapoint.numeric_restricts
    }
    
    # Test integer fields use value_int
    integer_restriction = num_restrictions["integer_field"]
    print(f"integer_field restriction: {integer_restriction}")
    
    # Check if it has value_int attribute and correct value
    if hasattr(integer_restriction, 'value_int') and integer_restriction.value_int is not None:
        print(f"✓ integer_field correctly uses value_int: {integer_restriction.value_int}")
        assert integer_restriction.value_int == 42, f"Expected 42, got {integer_restriction.value_int}"
    else:
        print(f"✗ integer_field does not use value_int properly")
        if hasattr(integer_restriction, 'value_float'):
            print(f"  Instead uses value_float: {integer_restriction.value_float}")
        return False
    
    # Test float fields use value_float
    float_restriction = num_restrictions["float_field"]
    print(f"float_field restriction: {float_restriction}")
    
    if hasattr(float_restriction, 'value_float') and float_restriction.value_float is not None:
        print(f"✓ float_field correctly uses value_float: {float_restriction.value_float}")
        assert abs(float_restriction.value_float - 3.14) < 0.001, f"Expected 3.14, got {float_restriction.value_float}"
    else:
        print(f"✗ float_field does not use value_float properly")
        return False
    
    # Test another integer field
    another_integer_restriction = num_restrictions["another_integer"]
    if hasattr(another_integer_restriction, 'value_int') and another_integer_restriction.value_int is not None:
        print(f"✓ another_integer correctly uses value_int: {another_integer_restriction.value_int}")
        assert another_integer_restriction.value_int == 100, f"Expected 100, got {another_integer_restriction.value_int}"
    else:
        print(f"✗ another_integer does not use value_int properly")
        return False
    
    # Test string restrictions still work
    string_restrictions = {
        restriction.namespace: restriction
        for restriction in datapoint.restricts
    }
    
    if "string_field" in string_restrictions:
        string_restriction = string_restrictions["string_field"]
        print(f"✓ string_field correctly handled: {string_restriction.allow_list}")
        assert string_restriction.allow_list == ["test_string"]
    else:
        print("✗ string_field not found in restrictions")
        return False
    
    print("✓ All tests passed! Integer metadata handling is working correctly.")
    return True

if __name__ == "__main__":
    try:
        success = test_integer_metadata_handling()
        if success:
            print("\n🎉 SUCCESS: The integer metadata fix is working correctly!")
            sys.exit(0)
        else:
            print("\n❌ FAILURE: The integer metadata fix is not working properly.")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
