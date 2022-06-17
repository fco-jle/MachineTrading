from itertools import combinations
import numpy as np

nums = [2, 7, 11, 15]
target = 9
cmb = list(combinations(nums, 2))
sums = np.array([x[0] + x[1] for x in cmb])
target_idx = np.where(sums == target)[0][0]
numbers = cmb[target_idx]
np.argmin(np.abs(np.array(nums)-numbers[0]))
target_indexes = [np.argmin(np.abs(np.array(nums)-numbers[i])) for i in range(len(numbers))]