import yaml

# Open the file and load the file
with open('wild-ti2i/wild-ti2i-real.yaml', 'r') as file:
    data = yaml.safe_load(file)

# Print the data to console

for item in data:
    print(item)
