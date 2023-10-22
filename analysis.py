import cv2
import numpy as np
from scipy.spatial.distance import euclidean


def remove_duplicates(centers, threshold=10):
    unique_centers = []
    for center in centers:
        is_duplicate = False
        for unique_center in unique_centers:
            if euclidean(center, unique_center) < threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_centers.append(center)
    return unique_centers


# Read color image and templates
image = cv2.imread('camground.png')
available_site = cv2.imread('available.png')
unavailable_site = cv2.imread('unavailable.png')

# Check if images are loaded correctly
if image is None or available_site is None or unavailable_site is None:
    print("One or more images were not loaded correctly. Please check the file paths.")
    exit(1)

# Template matching
res_A = cv2.matchTemplate(image, available_site, cv2.TM_CCOEFF_NORMED)
res_B = cv2.matchTemplate(image, unavailable_site, cv2.TM_CCOEFF_NORMED)

threshold = 0.9
loc_A = np.where(res_A >= threshold)
loc_B = np.where(res_B >= threshold)

coords_A = list(zip(*loc_A[::-1]))
coords_B = list(zip(*loc_B[::-1]))

# Dimensions of templates
w_A, h_A = available_site.shape[1], available_site.shape[0]
w_B, h_B = unavailable_site.shape[1], unavailable_site.shape[0]

# Initialize lists for centers and distances
centers_A = [(pt[0] + w_A // 2, pt[1] + h_A // 2) for pt in coords_A]
centers_A = remove_duplicates(centers_A)
centers_B = []
distances = []

# Debugging code for showing all detected sites
debug_image = image.copy()
for pt in coords_A:
    cv2.rectangle(debug_image, pt, (pt[0] + w_A, pt[1] + h_A), (0, 255, 0), 2)
for pt in coords_B:
    cv2.rectangle(debug_image, pt, (pt[0] + w_B, pt[1] + h_B), (0, 0, 255), 2)

# Calculate the distances
for center_A in centers_A:
    min_distance = float('inf')
    for pt_b in coords_B:
        center_B = (pt_b[0] + w_B // 2, pt_b[1] + h_B // 2)
        distance = euclidean(center_A, center_B)
        if distance < min_distance:
            min_distance = distance
    distances.append(min_distance)

# Debugging: print distances and centers
for i, (center, distance) in enumerate(zip(centers_A, distances)):
    print(f"Center {i+1}: {center}, Distance: {distance}")

sorted_indices = np.argsort(distances)[-5:]
print("Top 5:")
for i in sorted_indices:
    print(f"Center: {centers_A[i]}, Distance: {distances[i]}")

# Highlight the top 5 Type A entities with maximum distance
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.2
font_thickness = 3
top5_centers = [centers_A[i] for i in sorted_indices]

for i, center in enumerate(top5_centers):
    cv2.rectangle(image, (center[0] - w_A // 2, center[1] - h_A // 2),
                  (center[0] + w_A // 2, center[1] + h_A // 2), (255, 0, 0), 2)
    cv2.putText(image, f"Top {i+1}", (center[0] - 20, center[1] - 20),
                font, font_scale, (0, 0, 0), font_thickness)


cv2.imwrite('Debug_Detection.png', debug_image)  # Save debug image
cv2.imwrite('Top_5_Available_Sites.png', image)  # Save final image

# Show the debug image and final result
cv2.imshow('Campsite Availability Detection', debug_image)
cv2.imshow('Top 5 Available Sites', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
