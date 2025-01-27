import matplotlib.pyplot as plt
import numpy as np

##############################################################################
# 1) Parameters
##############################################################################
number_of_centroids = 25     # Number of random centroids
spacing_factor = 10          # Spacing between grid points
tolerance = 1e-5             # Convergence threshold
max_iterations = 500         # Safety limit on K-Means iterations
skip_the_pause = True

##############################################################################
# 2) Create the 100x100 grid of points
##############################################################################
# x-coordinates: 0, 10, 20, ..., 990
x = np.arange(100) * spacing_factor
# y-coordinates: 0, 10, 20, ..., 990
y = np.arange(100) * spacing_factor


# Create a meshgrid for the scatter plot (X, Y each shape (100, 100))
X, Y = np.meshgrid(x, y)

# Flatten into (10,000, 2)
points_array = np.column_stack([X.flatten(), Y.flatten()])

##############################################################################
# 3) Randomly pick initial centroids
##############################################################################
# Choose 'number_of_centroids' distinct points from the grid
unique_indices = np.random.choice(len(points_array),
                                  size=number_of_centroids,
                                  replace=False)
centroids = points_array[unique_indices].copy()

# Generate distinct colors for each centroid using a colormap
centroid_colors = plt.cm.rainbow(np.linspace(0, 1, number_of_centroids))

##############################################################################
# 4) Prepare Interactive Plot
##############################################################################
# Turn on interactive mode
plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))

def update_plot(title, points_color, centroids_current):
    """Helper function: clear the axes, plot points and centroids, refresh."""
    ax.clear()
    ax.set_title(title)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_xlim(-50, x[-1] + 50)
    ax.set_ylim(-50, y[-1] + 50)
    ax.set_aspect('equal', adjustable='box')
    
    # Plot the points (colored by centroid if available)
    ax.scatter(points_array[:, 0],
               points_array[:, 1],
               c=points_color,
               marker='.',
               s=10)
    
    # Plot centroids as large stars
    for i, (cx, cy) in enumerate(centroids_current):
        ax.scatter(cx, cy,
                   marker='*',
                   s=300,
                   color=centroid_colors[i],
                   edgecolor='black')
    
    plt.draw()
    if not skip_the_pause:
        print(f"{title} - Press any key or click in the figure to continue...")
        plt.waitforbuttonpress()
    else:
        print(f"{title} - Paused for {0.1} seconds...")
        plt.pause(0.1)


##############################################################################
# 5) Initial Plot: Show uncolored grid + random centroids
##############################################################################
# Color all points black initially
initial_point_colors = ['black'] * len(points_array)
update_plot("Initial Centroids", initial_point_colors, centroids)

##############################################################################
# 6) K-Means Iterations (Assignment -> Update)
##############################################################################
for iteration in range(1, max_iterations + 1):
    # --- (a) ASSIGNMENT STEP ---
    # Compute distances of every point to each centroid
    distance_matrix = np.linalg.norm(
        points_array[:, None, :] - centroids[None, :, :],
        axis=2
    )
    # Assign each point to nearest centroid index
    closest_centroid_indices = np.argmin(distance_matrix, axis=1)
    # Color each point based on assigned centroid
    point_colors = centroid_colors[closest_centroid_indices]
    
    # Show assignment
    update_plot(f"Iteration {iteration} - Assignment",
                point_colors,
                centroids)
    
    # --- (b) UPDATE STEP ---
    # Move each centroid to the mean of its assigned points
    new_centroids = []
    for i in range(number_of_centroids):
        # Extract points assigned to centroid i
        cluster_points = points_array[closest_centroid_indices == i]
        if len(cluster_points) > 0:
            new_location = cluster_points.mean(axis=0)
        else:
            # If no points assigned, keep the old centroid
            new_location = centroids[i]
        new_centroids.append(new_location)
    new_centroids = np.array(new_centroids)
    
    # Show updated centroid positions (the points remain the same colors from assignment)
    update_plot(f"Iteration {iteration} - Update",
                point_colors,
                new_centroids)
    
    # Compute shift to check convergence
    shifts = np.linalg.norm(new_centroids - centroids, axis=1)
    max_shift = np.max(shifts)
    centroids = new_centroids
    
    print(f"Iteration {iteration} - Max shift: {max_shift:.6f}")
    if max_shift < tolerance:
        print("Convergence reached!")
        break

# Turn off interactive mode; keep the final plot open
plt.ioff()
plt.show()
