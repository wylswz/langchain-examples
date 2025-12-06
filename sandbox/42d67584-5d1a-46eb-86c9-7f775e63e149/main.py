import math
import random

# Function to apply GELU activation
# GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
def gelu(x):
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))

# Step 1: Generate 100 values from a Gaussian distribution
values = [random.gauss(0, 1) for _ in range(100)]

# Step 2: Apply GELU activation function
activated_values = [gelu(x) for x in values]

# Step 3: Save the results to /results.txt
with open('/results_output.txt', 'w') as f:
    for value in activated_values:
        f.write(f"{value}\n")

# Step 4: Print the first 10 values
print(activated_values[:10])
