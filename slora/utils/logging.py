from datetime import datetime
import json

def print_with_timestamp(**kwargs):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"Logging|{current_time}|INFO", end="|")
    for key, value in kwargs.items():
        if isinstance(value, (dict, list)):
            print(f"{key}: {json.dumps(value, indent=None, ensure_ascii=False)}", end="|")
        else:
            print(f"{key}: {value}", end="|")
    print("")

# print_with_timestamp(
#     name="Alice",
#     age=25,
#     location="Singapore",
#     data_dict={"key1": "value1", "key2": "value2"},
#     data_list=["apple", "banana", "cherry"]
# )
