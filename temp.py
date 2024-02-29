import numpy as np
import matplotlib.pyplot as plt

# Generate x values from -5 to 5
x = np.linspace(-50, 50, 100)

# Calculate y values using tanh function
y = np.tanh(0.07*x)

# Plot tanh function
plt.plot(x, y)

# Add grid
plt.grid(True)

# Add labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-50, 50)

# Show plot
plt.show()
