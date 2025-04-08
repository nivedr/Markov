import pickle
import matplotlib.pyplot as plt

# Specify the path to your pickle file
pickle_file = 'val-loss-dump.pickle'

# Load the pickle file
with open(pickle_file, 'rb') as file:
    data = pickle.load(file)

print(data)

# Check if the loaded data is a dictionary
if isinstance(data, dict):
    print("Loaded dictionary:")
    for key, value in data.items():
        print(f"{key}: {value}")


    # Attempt to convert keys and values for visualization.
    # We will try to convert values to float if possible.
#    keys = []
#    numeric_values = []
#    for key, value in data.items():
#        try:
#            numeric_values.append(float(value))
#            keys.append(str(key))
#        except (ValueError, TypeError):
            # Skip keys that do not have a numeric value.
#            pass

#    if numeric_values:
#        plt.figure(figsize=(10, 6))
#        plt.bar(keys, numeric_values)
#        plt.title("Visualization of Dictionary Numeric Values")
#        plt.xlabel("Keys")
#        plt.ylabel("Values")
#        plt.xticks(rotation=45)
#        plt.tight_layout()
#        plt.show()
#    else:
#        print("No numeric values found for plotting.")
#else:
#    print("The pickle file does not hold a dictionary.")
