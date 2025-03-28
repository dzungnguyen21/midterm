import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_multi_color_histograms(image):
    """Compute color histograms in multiple color spaces."""
    histograms = {}

    # Convert to HSV and LAB color spaces
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Compute histograms
    hsv_hist = cv2.calcHist([hsv_image], [0, 1], None, [180, 256], [0, 180, 0, 256])
    lab_hist = cv2.calcHist([lab_image], [1, 2], None, [256, 256], [0, 256, 0, 256])

    # Normalize histograms
    cv2.normalize(hsv_hist, hsv_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(lab_hist, lab_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    histograms["hsv"] = hsv_hist
    histograms["lab"] = lab_hist

    return histograms

def compare_multi_color_histograms(hist1, hist2):
    """Compare multiple color histograms using Correlation & Intersection."""
    methods = [cv2.HISTCMP_CORREL, cv2.HISTCMP_INTERSECT]
    total_score = 0

    for color_space in hist1:
        if color_space in hist2:
            scores = [cv2.compareHist(hist1[color_space], hist2[color_space], method) for method in methods]
            total_score += np.mean(scores)

    return total_score / len(hist1)

def advanced_multi_scale_template_matching(main_image, template, scale_range=(0.1, 0.85), scale_steps=10):
    """Enhanced multi-scale template matching using edges & color histograms."""
    best_location, best_scale, best_edge_score, best_color_score, best_combined_score = None, 1.0, -np.inf, -np.inf, -np.inf
    scales = np.linspace(scale_range[0], scale_range[1], scale_steps)

    # Convert main image to grayscale
    main_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)

    for scale in scales:
        # Resize template
        resized_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        if resized_template.shape[0] == 0 or resized_template.shape[1] == 0:
            continue  # Skip invalid sizes

        # Convert template to grayscale
        template_gray = cv2.cvtColor(resized_template, cv2.COLOR_BGR2GRAY)

        # Edge detection (convert to uint8 to avoid errors)
        main_edges = cv2.Canny(main_gray, 50, 200)
        template_edges = cv2.Canny(template_gray, 50, 200)

        # Sobel gradients (convert to uint8 before matching)
        main_sobel_x = cv2.convertScaleAbs(cv2.Sobel(main_gray, cv2.CV_64F, 1, 0, ksize=3))
        main_sobel_y = cv2.convertScaleAbs(cv2.Sobel(main_gray, cv2.CV_64F, 0, 1, ksize=3))
        template_sobel_x = cv2.convertScaleAbs(cv2.Sobel(template_gray, cv2.CV_64F, 1, 0, ksize=3))
        template_sobel_y = cv2.convertScaleAbs(cv2.Sobel(template_gray, cv2.CV_64F, 0, 1, ksize=3))

        # Perform template matching
        edge_results = [
            cv2.matchTemplate(main_edges, template_edges, cv2.TM_CCOEFF_NORMED),
            cv2.matchTemplate(main_sobel_x, template_sobel_x, cv2.TM_CCOEFF_NORMED),
            cv2.matchTemplate(main_sobel_y, template_sobel_y, cv2.TM_CCOEFF_NORMED),
        ]

        # Compute color histograms
        template_hists = compute_multi_color_histograms(resized_template)
        h, w = resized_template.shape[:2]

        # Average edge results
        combined_edge_result = np.mean(edge_results, axis=0)

        # Iterate over possible matches
        for y in range(combined_edge_result.shape[0]):
            for x in range(combined_edge_result.shape[1]):
                current_edge_score = combined_edge_result[y, x]

                # Extract region of interest
                roi = main_image[y:y+h, x:x+w]
                if roi.shape[:2] != (h, w):
                    continue  # Skip mismatched sizes

                # Compute color histogram similarity
                roi_hists = compute_multi_color_histograms(roi)
                color_score = compare_multi_color_histograms(template_hists, roi_hists)

                # Weighted score combination
                combined_score = 0.7 * current_edge_score + 0.3 * color_score

                # Update best match
                if combined_score > best_combined_score:
                    best_location, best_scale = (x, y), scale
                    best_edge_score, best_color_score, best_combined_score = current_edge_score, color_score, combined_score

    return best_location, best_scale, best_edge_score, best_color_score, best_combined_score

# Load images
main_image = cv2.imread("Finding/image.png")
template = cv2.imread("Template/find/cake.png")

# Ensure images are loaded correctly
if main_image is None or template is None:
    raise ValueError("Error: Image or template not found. Check file paths.")

# Perform advanced template matching
match_location, match_scale, edge_score, color_score, combined_score = advanced_multi_scale_template_matching(main_image, template)

# Resize template to best scale
resized_template = cv2.resize(template, None, fx=match_scale, fy=match_scale, interpolation=cv2.INTER_LINEAR)
h, w = resized_template.shape[:2]

# Draw match on output image
output_image = main_image.copy()
cv2.rectangle(output_image, match_location, (match_location[0] + w, match_location[1] + h), (0, 255, 0), 2)

# Convert to RGB for Matplotlib
output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
template_rgb = cv2.cvtColor(resized_template, cv2.COLOR_BGR2RGB)

# Display results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(output_image_rgb)
plt.title(f"Detected Object\nCombined Score: {combined_score:.2f}")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(template_rgb)
plt.title(f"Template (Scale: {match_scale:.2f})")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Matching Scores")
plt.text(0.1, 0.8, f"Scale: {match_scale:.2f}", fontsize=10)
plt.text(0.1, 0.6, f"Edge Matching Score: {edge_score:.2f}", fontsize=10)
plt.text(0.1, 0.4, f"Color Histogram Score: {color_score:.2f}", fontsize=10)
plt.text(0.1, 0.2, f"Combined Score: {combined_score:.2f}", fontsize=10)
plt.axis('off')

plt.tight_layout()
plt.show()

# Print results
print(f"Match Location: {match_location}")
print(f"Match Scale: {match_scale:.2f}")
print(f"Edge Matching Score: {edge_score:.2f}")
print(f"Color Histogram Score: {color_score:.2f}")
print(f"Combined Score: {combined_score:.2f}")
