import pandas as pd

# Read each line of the JSON file as a separate JSON object
df = pd.read_json('train_cleaned.json', lines=True)

print(df.head(5))
print(df.shape)
print(df.columns)

# import json
# #
# # with open('train.json', 'r') as file:
# #     try:
# #         data = json.load(file)
# #         print(data)
# #     except json.JSONDecodeError as e:
# #         print(f"Error decoding JSON: {e}")
#
#
# #### creating a modified file as the original train.json file has some error
# #### [Error decoding JSON: Expecting value: line 2 column 1 (char 2)]
#
# # import json
# #
# # # Step 1: Open and read the original file, then replace the content
# # with open('train.json', 'r') as file:
# #     try:
# #         # Read each line as a separate JSON object
# #         data = [json.loads(line) for line in file]
# #
# #         # Modify the data (Example: adding a new field here, modify as needed)
# #         for item in data:
# #             item['new_field'] = 'some_value'  # Example: adding a new field
# #
# #         # Step 2: Save the modified data into a new file (train_modified.json)
# #         with open('train_modified.json', 'w') as outfile:
# #             json.dump(data, outfile, indent=4)  # Saving the modified data to a new file
# #
# #         print("Modified file saved as 'train_modified.json'.")
# #
# #     except json.JSONDecodeError as e:
# #         print(f"Error decoding JSON: {e}")
# #
# # # Step 3: Open the newly saved file to check the shape of the content
# # with open('train_modified.json', 'r') as file:
# #     try:
# #         # Read the modified data
# #         data = [json.loads(line) for line in file]
# #
# #         # Calculate rows and columns
# #         if data:
# #             rows = len(data)  # Number of rows (items in the list)
# #             columns = len(data[0].keys())  # Number of columns (keys in the first dictionary)
# #             print(f"Rows: {rows}, Columns: {columns}")
# #         else:
# #             print("No data found in the file.")
# #
# #     except json.JSONDecodeError as e:
# #         print(f"Error decoding JSON: {e}")
# #


# import pandas as pd

# Load the modified JSON file
df = pd.read_json('train_modified.json')

# # Remove the 'new_field' column
# if 'new_field' in df.columns:
#     df = df.drop(columns=['new_field'])
#
# # Save the cleaned DataFrame back to a new JSON file
# df.to_json('train_cleaned.json', orient='records', lines=True)
#
# # Print the shape and columns to verify
# print(df.shape)
# print(df.columns)
