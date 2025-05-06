import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

# Parameters
num_points = 6
radius = 1
angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, num_points, endpoint=False)

# Coordinates of points
x = radius * np.cos(angles)
y = radius * np.sin(angles)

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')

# Plot points
ax.plot(x, y, 'o', color='black')

labels = ["discriminator\ndiscovers",
          "imitator\nimitates",
          "discriminator\npenalizes",
          "imitator\nadapts",
          "discriminator\nforgets",
          "imitator\nforgets"]

# Draw curved arrows and labels
for i in range(num_points):
    start = (x[i], y[i])
    end = (x[(i + 1) % num_points], y[(i + 1) % num_points])
    
    # Draw curved arrow (arc between points)
    arrow = FancyArrowPatch(end, start,
                            connectionstyle="arc3,rad=0.3",
                            arrowstyle='<-',
                            mutation_scale=20,
                            linewidth=5,
                            color='blue')
    ax.add_patch(arrow)
    
    # Midpoint for label
    mid_x = (start[0] + end[0]) / 2
    mid_y = (start[1] + end[1]) / 2

    # Shift label slightly outward from the center to avoid overlap
    center_x, center_y = 0, 0
    dx = mid_x - center_x
    dy = mid_y - center_y
    norm = np.sqrt(dx**2 + dy**2)
    offset = 0.15  # adjust as needed

    label_x = mid_x + (dx / norm) * offset
    label_y = mid_y + (dy / norm) * offset
    label = labels[i]
    ax.text(label_x, label_y, label, fontsize=9, ha='center', va='center', backgroundcolor='white')

# Hide axes
ax.axis('off')
plt.title(r"ℤ₆", fontsize=16)
plt.savefig("diagram_Z6.png", bbox_inches='tight')
