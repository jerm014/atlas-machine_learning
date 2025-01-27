import matplotlib.pyplot as plt
import numpy as np

##############################################################################
# 1) Parameters
##############################################################################
number_of_centroids = 40      # Number of random centroids
spacing_factor = 1           # Spacing between grid points
tolerance = 1e-5             # Convergence threshold
max_iterations = 100          # Safety limit on K-Means iterations
skip_the_pause = True        # If True, auto-advance; if False, waitforbuttonpress

##############################################################################
# 2) Create the 100x100x100 grid of points (BEWARE of 1e6 points)
##############################################################################
x = np.arange(40) * spacing_factor   # e.g., [0, 1, 2, ..., 99]
y = np.arange(40) * spacing_factor
z = np.arange(40) * spacing_factor

# X, Y, Z each shape (100, 100, 100)
X, Y, Z = np.meshgrid(x, y, z, indexing='xy')

# Flatten into an array of shape (1,000,000, 3)
points_array = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

##############################################################################
# 3) Randomly pick initial centroids from the grid
##############################################################################
unique_indices = np.random.choice(len(points_array),
                                  size=number_of_centroids,
                                  replace=False)
centroids = points_array[unique_indices].copy()

##############################################################################
# 4) Colors
##############################################################################
# Generate distinct colors for each centroid using a colormap
centroid_colors = plt.cm.rainbow(np.linspace(0, 1, number_of_centroids))

##############################################################################
# 5) Prepare Interactive 3D Plot
##############################################################################
plt.ion()  # Turn on interactive mode

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

def update_plot(title, point_colors, current_centroids):
    """Clear the 3D axes, plot points and centroids, refresh."""
    ax.clear()
    ax.set_title(title)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    
    # Optional: adjust viewing angle or axis limits
    # ax.set_xlim(0, x[-1])
    # ax.set_ylim(0, y[-1])
    # ax.set_zlim(0, z[-1])

    # Plot *all* points in 3D:
    # WARNING: 1 million points can be very slow to render!
    ax.scatter(points_array[:, 0],
               points_array[:, 1],
               points_array[:, 2],
               c=point_colors,
               marker='.',
               s=1,
               alpha=0.6)  # alpha helps see overlapping points

    # Plot centroids as stars
    for i, (cx, cy, cz) in enumerate(current_centroids):
        ax.scatter(cx, cy, cz,
                   marker='*',
                   s=200,
                   color=centroid_colors[i],
                   edgecolor='black')

    plt.draw()

    if not skip_the_pause:
        print(f"{title} - Press any key or click in the figure to continue...")
        plt.waitforbuttonpress()
    else:
        print(f"{title} - Pausing for 0.1 seconds...")
        plt.pause(0.1)

##############################################################################
# 6) Show initial centroids (all points in black)
##############################################################################
initial_point_colors = ['black'] * len(points_array)
update_plot("Initial Centroids", initial_point_colors, centroids)

##############################################################################
# 7) K-Means Iterations (Assignment -> Update)
##############################################################################
for iteration in range(1, max_iterations + 1):
    # --- (a) ASSIGNMENT STEP ---
    # Compute distances of every point to each centroid
    distance_matrix = np.linalg.norm(points_array[:, None, :] - centroids[None, :, :],
                                     axis=2)
    # Assign each point to nearest centroid index
    closest_centroid_indices = np.argmin(distance_matrix, axis=1)
    # Color each point based on assigned centroid
    point_colors = centroid_colors[closest_centroid_indices]
    
    # Show assignment
    update_plot(f"Iteration {iteration} - Assignment", point_colors, centroids)
    
    # --- (b) UPDATE STEP ---
    # Move each centroid to the mean of its assigned points
    new_centroids = []
    for i in range(number_of_centroids):
        cluster_points = points_array[closest_centroid_indices == i]
        if len(cluster_points) > 0:
            new_location = cluster_points.mean(axis=0)
        else:
            new_location = centroids[i]
        new_centroids.append(new_location)
    new_centroids = np.array(new_centroids)
    
    # Show updated centroids (same color assignment for now)
    update_plot(f"Iteration {iteration} - Update", point_colors, new_centroids)
    
    # Convergence check
    shifts = np.linalg.norm(new_centroids - centroids, axis=1)
    max_shift = np.max(shifts)
    centroids = new_centroids
    
    print(f"Iteration {iteration} - Max shift: {max_shift:.6f}")
    if max_shift < tolerance:
        print("Convergence reached!")
        break

plt.ioff()  # Turn off interactive mode
plt.show()
