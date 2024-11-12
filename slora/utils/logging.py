from datetime import datetime
import json

def print_with_timestamp(**kwargs):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for key, value in kwargs.items():
        if isinstance(value, (dict, list)):
            print(f"{key}: {json.dumps(value, indent=4, ensure_ascii=False)}")
        else:
            print(f"{key}: {value}")
    
    print(f"Current Time: {current_time}")

# print_with_timestamp(
#     name="Alice",
#     age=25,
#     location="Singapore",
#     data_dict={"key1": "value1", "key2": "value2"},
#     data_list=["apple", "banana", "cherry"]
# )
