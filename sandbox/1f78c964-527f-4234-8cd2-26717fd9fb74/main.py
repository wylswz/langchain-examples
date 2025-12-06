import numpy as np

# Step 1: Generate 100 values from a Gaussian distribution
values = np.random.normal(loc=0, scale=1, size=100)

# Step 2: Define the GELU activation function
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

# Apply GELU to the generated values
activated_values = gelu(values)

# Step 3: Save the results to /results.txt
with open('/data/results.txt', 'w') as f:
    for value in activated_values:
        f.write(f"{value}\n")

# Step 4: Print the first 10 values
print(activated_values[:10])
