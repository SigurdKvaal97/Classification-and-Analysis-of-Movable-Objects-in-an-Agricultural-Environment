import cv2
import os
import json

# Global variable to store points
points = []

# Mouse callback function to capture points
def draw_circle(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Image Viewer", param)

def display_images(image_folder, output_json, marked_image_folder, video, verb, noun):
    global points

    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('jpg', 'jpeg', 'png'))]
    image_files.sort()  # Sort the files by name

    if not image_files:
        print("No images found in the specified folder.")
        return

    # Create the marked image folder if it doesn't exist
    os.makedirs(marked_image_folder, exist_ok=True)

    image_data = []
    current_index = 0

    while True:
        image_path = os.path.join(image_folder, image_files[current_index])
        image = cv2.imread(image_path)
        original_image = image.copy()  # Keep a copy of the original image

        if image is None:
            print(f"Could not open or find the image: {image_path}")
            break

        height, width, _ = image.shape

        # Reset points for the new image
        points = []
        
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f"Image {current_index + 1}/{len(image_files)}")
        plt.show()

        cv2.imshow("Image Viewer", image)
        cv2.setMouseCallback("Image Viewer", draw_circle, image)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                current_index = (current_index - 1) % len(image_files)
                break

            elif key == ord('d'):
                current_index = (current_index + 1) % len(image_files)
                break

            elif key == ord('x'):
                current_index = (current_index + 100) % len(image_files)
                break

            elif key == ord('z'):
                current_index = (current_index - 100) % len(image_files)
                break
            elif key == ord(' '):

                prefix = video + "_"
                new_im_name = prefix + image_files[current_index]

                # Save current points and move to the next image
                image_info = {
                    "image": [new_im_name],
                    "shape": [height, width],
                    "verb": verb,  # Placeholder value
                    "noun": noun,  # Placeholder value
                    "points": points
                }
                image_data.append(image_info)
                


                # Save the original image without dots
                marked_image_path = os.path.join(marked_image_folder, new_im_name)
                
                cv2.imwrite(marked_image_path, original_image)

                current_index = (current_index + 1) % len(image_files)
                break
            elif key == ord('q'):
                cv2.destroyAllWindows()
                with open(output_json, 'w') as f:
                    json.dump({"train_images": image_data}, f, indent=4)
                return

    cv2.destroyAllWindows()
    with open(output_json, 'w') as f:
        json.dump({"train_images": image_data}, f, indent=4)

if __name__ == "__main__":

    #needs to be adjusted
    #"verbs": ["pick", "move", "cut", "put"], 
    #"nouns": ["plant", "berry"],

    video="M01_01"
    verb=0
    noun= 0
    #---------------------

    # example: "M01"
    sect = video[0:3]

    image_folder = f"/home/filip_lund_andersen/MasterData/frames_rgb_flow/rgb/train/{sect}/{video}"  # Replace with your image folder path
    output_json = f"/home/filip_lund_andersen/interaction-hotspots-master_masterdata/validation/val_data_{video}.json"  # Output JSON file
    marked_image_folder = f"/home/filip_lund_andersen/interaction-hotspots-master_masterdata/validation/val_im{video}"  # Folder to save marked images
    display_images(image_folder, output_json, marked_image_folder, video, verb, noun)
