import sys
sys.path.append('.')

import os
import pickle
import numpy as np

from helper_functions.utils import get_data

# def closest_point_index(points, ref_point=[0.0, 0.0], threshold=0.15):
#     """
#     Given an array of 8 numbers (4 points of (x, y)) and a reference point,
#     return the index (0-based) of the point that is within the given threshold.
#     If multiple points are within threshold, return the index of the closest one.
#     If no point is within threshold, return None.
#     """
#     # Reshape the array into (4, 2)
#     pts = np.array(points).reshape(-1, 2)
    
#     # Compute Euclidean distances from each point to the reference point
#     distances = np.linalg.norm(pts - np.array(ref_point), axis=1)
    
#     # Find indices of points within the threshold distance
#     indices_within = np.where(distances <= threshold)[0]
    
#     if indices_within.size == 0:
#         return 4  # No point is close enough
    
#     # If more than one point qualifies, pick the one with the smallest distance
#     closest_idx = indices_within[np.argmin(distances[indices_within])]
#     # closest_idx = np.argmin(distances)
#     return closest_idx

# def closest_point_index(points, ref_point=[0.0, 0.0]):
#     """
#     Given an array of 8 numbers (4 points of (x, y)) and a reference point,
#     return the index (0-based) of the point that is within the given threshold.
#     If multiple points are within threshold, return the index of the closest one.
#     If no point is within threshold, return None.
#     """
#     # Reshape the array into (4, 2)
#     pts = np.array(points).reshape(-1, 2)
    
#     # Compute Euclidean distances from each point to the reference point
#     distances = np.linalg.norm(pts - np.array(ref_point), axis=1)
#     closest_idx = np.argmin(distances)
#     return closest_idx

def closest_point_index(points, threshold=0.15):
    """
    Given an array of 8 numbers (4 points of (x, y)) and a reference point,
    return the index (0-based) of the point that is within the given threshold.
    If multiple points are within threshold, return the index of the closest one.
    If no point is within threshold, return None.
    """
    # Find indices of points within the threshold distance
    indices_within = np.where(points <= threshold)[0]
    
    if indices_within.size == 0:
        return 4  # No point is close enough
    
    # If more than one point qualifies, pick the one with the smallest distance
    closest_idx = indices_within[np.argmin(points[indices_within])]
    # closest_idx = np.argmin(distances)
    return closest_idx


# def closest_point_index(points):
#     assert len(points) == 4
#     return np.argmin(points)



if __name__ == "__main__":
    data = get_data("data/reacher/data.pkl")
    transitions = data.observations
    transitions = np.squeeze(transitions, axis=1)


    data_path = 'data/reacher/states.pkl'
    with open(data_path, 'wb') as f:
        pickle.dump(transitions, f)
    print(f"Collected {len(transitions)} transitions and saved to {data_path}")

    class_prob = []
    for transition in transitions:
        sliced_data = transition[-4:]
        cls = closest_point_index(sliced_data)
        class_prob.append(cls)

    counts = {}
    for element in class_prob:
        counts[element] = counts.get(element, 0) + 1

    # Print each element and its count
    for element, count in counts.items():
        print(f"{element}: {count}")

    # Number of unique categories
    num_categories = np.max(class_prob) + 1
    # One-hot encode
    class_prob = np.eye(num_categories)[class_prob]

    data_path = 'data/reacher/class_prob.pkl'
    with open(data_path, 'wb') as f:
        pickle.dump(class_prob, f)
    print(f"Collected {len(class_prob)} transitions and saved to {data_path}") 


