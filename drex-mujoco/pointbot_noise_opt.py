
import numpy as np
from envs.pointbot import PointBotTeacher
import itertools
import pickle
import polytope as pc 

"""
Computes constraint volume of feature expectation vectors by sorting them
based on their noise magnitude and then doing some computations. Currently
via MC sampling, constraint_volume is on [0, 1] (percent of sampled points
that satisfy the constraints)
"""
def compute_constraint_volume(feature_exps, N_samples=10000000, volume_method="chebyshev_ball"):
	# First sort feature_exps by their noise value
	sorted_feature_exps = sorted(feature_exps, key=lambda x: x[1])
	# Generate constraint matrix
	constraint_matrix = []
	for i in range(len(sorted_feature_exps) - 1):
		for j in range(i+1, len(sorted_feature_exps)):
			constraint_matrix.append(sorted_feature_exps[i][0] - sorted_feature_exps[j][0])

	A = np.array(constraint_matrix)
	b = np.zeros(A.shape[0])

	if volume_method == "monte-carlo":
		# Monte-Carlo estimate of constraint volume --> (Assume weights on [-1, 1]^k)
		monte_carlo_samples = 2*np.random.random_sample((len(b), N_samples)) - 1
		constraint_vals = A.dot(monte_carlo_samples)
		max_col_vals = np.max(constraint_vals, axis=0) # Compute max value in each column
		# Find for how often the max column value is negative (ie. it satisfies all constraints)
		constraint_volume = float(len(max_col_vals[max_col_vals < 0]))/float(N_samples)
	elif volume_method == "chebyshev_ball":
		# Compute polytope enforcing every ||w||_1 <= 1
		new_constraints = np.vstack((np.eye(A.shape[-1]), -np.eye(A.shape[-1])))
		A_new = np.vstack((A, new_constraints))
		b_new = np.append(b, [1]*2*A.shape[-1])
		# print(A_new)
		# print(b_new)
		p = pc.Polytope(A_new, b_new)
		constraint_volume = p.chebR
		# print(constraint_volume)

	return constraint_volume

"""
Searches over noise parameters for the one which minimizes constraint volume
"""
def optimize_noise_grid_search(feature_exps, noise_low, noise_high, discretization, num_samples=10):
	grid_points = [np.linspace(noise_low[i], noise_high[i], discretization[i]) for i in range(len(discretization))]
	print("GRID POINTS: ", grid_points)
	if len(grid_points) == 1:
		noise_grid = grid_points[0]
	else:
		noise_grid = list(itertools.product(*grid_points))

	min_constraint_volume = np.inf
	min_constraint_volume_feature_exp = feature_exps[0]

	for i, noise_val in enumerate(noise_grid):
		print("Noise Value: ", i)
		# Sample from policy with this noise_val
		demos = [teacher.get_rollout(noise_val) for _ in range(num_samples)] 
		# Compute feature expectation
		feature_exp = (np.mean(np.array([d["features"] for d in demos]), axis=(0,1)), noise_val)
		# Compute constraint volume when adding these feature expectations to feature_exps
		new_feature_exps = feature_exps + [feature_exp]
		constraint_volume = compute_constraint_volume(new_feature_exps)

		if constraint_volume < min_constraint_volume:
			min_constraint_volume = constraint_volume
			min_constraint_volume_feature_exp = feature_exp

	print(min_constraint_volume_feature_exp)

	# Return the optimized noise value and the corresponding feature expectations
	return min_constraint_volume_feature_exp, min_constraint_volume

def do_training(teacher, num_training_iters, iter_samples, num_bc_trajs):
	# Get demos from teacher
	demos = [teacher.get_rollout() for _ in range(iter_samples)] 
	# Get feature expectation of demos
	feature_exps = [(np.mean(np.array([d["features"] for d in demos]), axis=(0,1)), 0)]

	for i in range(num_training_iters):
    	# Get optimized noise parameter and corresponding feeature expectation
		opt_feature_exp, constraint_volume = optimize_noise_grid_search(feature_exps, [0], [0.75], [10])
    	# Add new optimized feature expectation to list
		feature_exps.append(opt_feature_exp)
		print("Iteration: ", i, " Constraint Volume: ", constraint_volume)

	print("FEATURE EXPS ", feature_exps)

	# Save information to train reward function
	get_rollouts(feature_exps, iter_samples, num_bc_trajs)

def get_rollouts(feature_exps, iter_samples, num_bc_trajs):
	# Sample rollouts from the learned noise values and generate dataset for TREX
	noise_vals = [f[1] for f in feature_exps]
	noise_injected_rollouts = []
	for n in noise_vals:
		noise_n_rollouts = []
		for _ in range(iter_samples):
			rollout_info = teacher.get_rollout(n)
			noise_n_rollouts.append( (rollout_info["features"], rollout_info["actions"], rollout_info["rewards"]) )

		noise_injected_rollouts.append( (n, noise_n_rollouts))

	with open('learned_noise_rollouts.pkl','wb') as f:
		pickle.dump(noise_injected_rollouts, f)

	suboptimal_demo_rollouts = []
	for _ in range(num_bc_trajs):
		rollout_info = teacher.get_rollout(0)
		suboptimal_demo_rollouts.append( (rollout_info["features"], rollout_info["actions"], rollout_info["rewards"]) )

	with open('no_noise_rollouts.pkl','wb') as g:
		pickle.dump(suboptimal_demo_rollouts, g)

if __name__ == "__main__":
    teacher = PointBotTeacher()
    teacher.env.set_mode(1)
    do_training(teacher, num_training_iters=50, iter_samples=10, num_bc_trajs = 30)
