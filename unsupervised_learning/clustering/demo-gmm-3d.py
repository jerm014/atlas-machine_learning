import numpy as np
import matplotlib.pyplot as plt

##############################################################################
# User Parameters
##############################################################################
num_data_points = 2000      # Number of (random) 3D points
number_of_components = 4    # Number of Gaussian components (clusters)
max_iterations = 150         # Max EM iterations
tolerance = 1e-4            # Convergence threshold
skip_the_pause = True      # If False, waits for keypress each iteration
pause_seconds = 0.1         # If skip_the_pause=True, how long to pause

##############################################################################
# Data Generation: uneven 3D distribution (Gaussian blobs)
##############################################################################
def generate_uneven_distribution_3d():
    """
    Sample points from multiple 3D Gaussian clusters (blobs) 
    with different means and standard deviations.
    """
    cluster_centers = np.array([
        [10, 10, 10],
        [50, 50, 50],
        [80, 10, 70],
        [15, 80, 40]
    ])
    # If you want exactly 'number_of_components' centers, adapt accordingly.

    # Example: number of points in each cluster
    points_per_cluster = [500, 700, 400, 400]
    # Example: standard deviation for each cluster
    cluster_stds = [5, 10, 8, 6]

    all_points = []
    for (center, n_points, std) in zip(cluster_centers, points_per_cluster, cluster_stds):
        pts = np.random.normal(loc=center, scale=std, size=(n_points, 3))
        all_points.append(pts)
    return np.vstack(all_points)

points_array = generate_uneven_distribution_3d()
# Alternatively: 
# points_array = 100 * np.random.rand(num_data_points, 3)  # uniform in [0,100]^3

N = len(points_array)
dim = 3  # 3D

##############################################################################
# Helper: Multivariate Gaussian PDF
##############################################################################
def gaussian_pdf(X, mean, cov):
    """
    Compute the multivariate Gaussian density for each point in X.

    X: shape (N, D)
    mean: shape (D,)
    cov: shape (D, D)
    Returns: shape (N,) array of probabilities p(x_i).
    """
    # Ensure X is (N, D)
    diff = X - mean  # shape (N, D)
    # Mahalanobis distance = diff * inv(cov) * diff^T
    # We use np.einsum for vectorized computations
    inv_cov = np.linalg.inv(cov)
    # (N, D) @ (D, D) -> (N, D), then element-wise multiply by diff (N, D), sum across D -> (N,)
    exponent = np.einsum('ij,jk,ik->i', diff, inv_cov, diff)
    det_cov = np.linalg.det(cov)
    norm_const = np.sqrt((2 * np.pi) ** dim * det_cov)
    return np.exp(-0.5 * exponent) / norm_const

##############################################################################
# Initialize GMM Parameters
##############################################################################
# 1) Weights (pi_k): start uniformly
weights = np.ones(number_of_components) / number_of_components

# 2) Means (mu_k): pick random points from dataset
rand_indices = np.random.choice(N, size=number_of_components, replace=False)
means = points_array[rand_indices].copy()

# 3) Covariances (Sigma_k): initialize to identity or small random near identity
covariances = np.array([np.eye(dim) for _ in range(number_of_components)])
# Or for a rougher start: covariances = np.array([np.cov(points_array.T)] * number_of_components)

##############################################################################
# Visualization Setup (3D)
##############################################################################
plt.ion()
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Distinct colors for each component
colors = plt.cm.rainbow(np.linspace(0, 1, number_of_components))

def update_plot(title, cluster_assignments, current_means):
    """
    Clear the 3D axes, plot data points color-coded by their *hard* cluster,
    and plot the GMM means as stars.
    """
    ax.clear()
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Hard assignment for coloring: pick argmax responsibility
    point_colors = [colors[cid] for cid in cluster_assignments]

    ax.scatter(points_array[:, 0],
               points_array[:, 1],
               points_array[:, 2],
               c=point_colors,
               marker='.',
               s=3,
               alpha=0.5)

    # Plot means as stars
    for i, (mx, my, mz) in enumerate(current_means):
        ax.scatter(mx, my, mz,
                   marker='*',
                   s=300,
                   color=colors[i],
                   edgecolor='black')

    plt.draw()
    if skip_the_pause:
        plt.pause(pause_seconds)
    else:
        print(f"{title} - Press any key / click in figure to proceed.")
        plt.waitforbuttonpress()

##############################################################################
# EM Algorithm for GMM
# We'll iterate and show each step until convergence or max_iterations.
##############################################################################

# We store responsibilities for each point for each component: shape (N, K)
responsibilities = np.zeros((N, number_of_components))

def compute_log_likelihood():
    """
    Compute the log-likelihood of the current model parameters
    given all data: sum over i=1..N of log( sum_{k} pi_k * N(x_i|mu_k,Sigma_k) ).
    """
    # For each component, compute p(x_i|mu_k, Sigma_k)
    pdf_matrix = np.zeros((N, number_of_components))
    for k in range(number_of_components):
        pdf_matrix[:, k] = gaussian_pdf(points_array, means[k], covariances[k])
    weighted_pdfs = pdf_matrix * weights  # shape (N, K)
    # sum across K for each point
    sum_over_k = np.sum(weighted_pdfs, axis=1)
    # avoid log(0)
    sum_over_k[sum_over_k < 1e-15] = 1e-15
    ll = np.sum(np.log(sum_over_k))
    return ll

old_log_likelihood = -np.inf

for iteration in range(1, max_iterations + 1):
    # ===================== E-STEP =====================
    # Compute unnormalized responsibilities = pi_k * N(x_i|mu_k, Sigma_k)
    for k in range(number_of_components):
        resp_k = weights[k] * gaussian_pdf(points_array, means[k], covariances[k])
        responsibilities[:, k] = resp_k

    # Normalize to get actual responsibilities
    # shape (N, K)
    sum_resps = np.sum(responsibilities, axis=1).reshape(-1, 1)
    # Avoid division by zero:
    sum_resps[sum_resps < 1e-15] = 1e-15
    responsibilities /= sum_resps

    # For color-coding each point by its "dominant" cluster:
    hard_assignments = np.argmax(responsibilities, axis=1)

    update_plot(f"Iteration {iteration} - After E-step", hard_assignments, means)

    # ===================== M-STEP =====================
    # N_k = sum of responsibilities for eac
