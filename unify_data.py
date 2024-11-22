import json

def transform_tool_selection_record(record):
    """Transform a tool selection record to the unified format"""
    return {
        "id": record["id"],
        "type": "tool_selection",
        "user_query": record["user_query"],
        "ground_truth": {
            "function_call": record["ground_truth"]["function_call"],
            "text": None,
            "expected_response_type": "function_call"
        }
    }

def transform_other_record(record, type_name):
    """Transform a non-tool record to the unified format"""
    return {
        "id": record["id"],
        "type": type_name,
        "user_query": record["user_query"],
        "ground_truth": {
            "function_call": None,
            "text": record["ground_truth"]["text"],
            "expected_response_type": "text"
        }
    }

def combine_datasets(output_path):
    # Define dataset paths and their types
    datasets = [
        ("datasets/test_tool_selection.json", "tool_selection"),
        ("datasets/test_no_tool.json", "no_tools"),
        ("datasets/test_not_supported.json", "not_supported"),
        ("datasets/test_error.json", "error"),
        ("datasets/test_clarifying.json", "clarifying")
    ]
    
    unified_records = []
    dataset_counts = {}
    
    # Process each dataset
    for path, type_name in datasets:
        with open(path, 'r') as f:
            data = json.load(f)
            dataset_counts[type_name] = len(data)
            
            for record in data:
                if type_name == "tool_selection":
                    unified_records.append(transform_tool_selection_record(record))
                else:
                    unified_records.append(transform_other_record(record, type_name))
    
    # Save combined dataset
    with open(output_path, 'w') as f:
        json.dump(unified_records, f, indent=2)
    
    # Print statistics
    print(f"Combined dataset saved to {output_path}")
    print(f"Total records: {len(unified_records)}")
    print("\nRecords per type:")
    for type_name, count in dataset_counts.items():
        print(f"- {type_name}: {count}")

if __name__ == "__main__":
    combine_datasets("datasets/unified_test_dataset.json")