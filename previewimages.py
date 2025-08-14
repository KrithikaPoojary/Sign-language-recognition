import cv2
import os

folder = "data/hello"  # Or use the full path if needed

for i in range(1, 6):
    img_path = os.path.join(folder, f"{i}.jpg")
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            cv2.imshow("Image Preview", img)
            cv2.waitKey(500)
        else:
            print(f"Unable to read {img_path}")
    else:
        print(f" File not found: {img_path}")

cv2.destroyAllWindows()
