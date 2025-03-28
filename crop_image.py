import cv2

# Open the image
image = cv2.imread("Finding/image.png")
template = cv2.imread("Template/boat.png")


# Ensure the crop box is within image bounds
h, w = image.shape[:2]
x_start, y_start = 242, 98
x_end, y_end = x_start + w, y_start + h

# Crop the image
cropped_image = image[y_start:y_end, x_start:x_end]

# Save and display the cropped image
cv2.imwrite("cropped_image.jpg", cropped_image)
cv2.imshow("Cropped Image", cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
