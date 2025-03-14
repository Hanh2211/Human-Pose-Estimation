import os
import random
import tkinter as tk
from tkinter import filedialog
import time
import cv2
import matplotlib.pyplot as plt
import torch
from PIL import Image
from matplotlib import patches
from torchvision.models.detection import keypointrcnn_resnet50_fpn

# Define your device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SCORE_THRESHOLD = 0.9
KEYPOINT_THRESHOLD = 0.9
connections = [(0, 1), (0, 2), (1, 3), (2, 4), (6, 5), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (12, 11),
               (11, 13), (13, 15), (12, 14), (14, 16)]


def keypoints_on_webcam(model):
    cap = cv2.VideoCapture(0)

    # Set the desired frame width and height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Continuously capture frames from the webcam and process them
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_tensor, _ = transform_val(pil_image)

        # Move the image tensor to the specified device and add a batch dimension
        output = model(image_tensor.to(device).unsqueeze(0))[0]
        output = {k: v.to("cpu") for k, v in output.items()}

        keypoints_on_cv2(frame, output, connections, SCORE_THRESHOLD, KEYPOINT_THRESHOLD)

        cv2.imshow('Webcam', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def keypoints_on_video(model, file):
    cap = cv2.VideoCapture(file)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    output_path = 'output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Loop through each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_tensor, _ = transform_val(pil_image)

        # Move the image tensor to the specified device and add a batch dimension
        output = model(image_tensor.to(device).unsqueeze(0))[0]
        output = {k: v.to("cpu") for k, v in output.items()}

        # Add dots to the frame
        frame_with_dots = keypoints_on_cv2(frame, output, connections, SCORE_THRESHOLD, KEYPOINT_THRESHOLD)

        # Write the frame to the output video
        out.write(frame_with_dots)

        # Display the frame
        cv2.imshow('Frame', frame_with_dots)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Output video saved to {output_path}")


def keypoints_on_image(model, file):
    start_time = time.time()  # Record start time
    image = Image.open(file)
    image_tensor, _ = transform_val(image)

    # Move the image tensor to the specified device and add a batch dimension
    output = model(image_tensor.to(device).unsqueeze(0))[0]
    output = {k: v.to("cpu") for k, v in output.items()}

    # Extract relevant information from the output
    boxes = output["boxes"]
    scores = output["scores"]
    keypoints = output["keypoints"]
    keypoints_scores = output["keypoints_scores"]

    # Create a subplot for displaying the image
    fig, ax = plt.subplots()
    ax.imshow(image)

    # Iterate through detected objects and draw bounding boxes and keypoints
    for box, score, keypoints, kpts_scores in zip(boxes, scores, keypoints, keypoints_scores):
        if score > SCORE_THRESHOLD:
            # Draw a bounding box around the detected object
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1,
                                     edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            # Draw lines connecting keypoints if their scores are above the threshold
            for p1, p2 in connections:
                if kpts_scores[p1] > KEYPOINT_THRESHOLD and kpts_scores[p2] > KEYPOINT_THRESHOLD:
                    # Generate a random color for each line
                    color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
                    x1, y1, _ = keypoints[p1]
                    x2, y2, _ = keypoints[p2]
                    ax.plot([x1, x2], [y1, y2], marker='o', linestyle='-', color=color, linewidth=1, markersize=2)

    plt.show()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

def select_file():
    """Open a file dialog to select a file."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select a file",
        filetypes=[
            ("Image Files", "*.png *.jpg *.jpeg *.bmp"),
            ("Video Files", "*.mp4 *.avi *.mov"),
            ("All Files", "*.*")
        ]
    )
    root.destroy()
    return file_path


def display_menu():
    """Display the main menu and get user input."""
    print("\n===== Keypoint Detection Menu =====")
    print("1. Use webcam")
    print("2. Process an image")
    print("3. Process a video")
    print("4. Exit")

    choice = input("Enter your choice (1-4): ")
    return choice.strip()


def select_file():
    """Allow user to input a file path."""
    print("\nEnter the file path (absolute or relative):")
    file_path = input("> ").strip()

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return None

    return file_path


def display_menu():
    """Display the main menu and get user input."""
    print("\n===== Keypoint Detection Menu =====")
    print("1. Use webcam")
    print("2. Process an image")
    print("3. Process a video")
    print("4. Exit")

    choice = input("Enter your choice (1-4): ")
    return choice.strip()

model = keypointrcnn_resnet50_fpn().to(device)
try:
  model.load_state_dict(torch.load("e42_b8_lr0.02_m0.9.pth", map_location=device))
  print("Model loaded successfully!")
except Exception as e:
  print(f"Error loading model: {e}")


def main():
    # Load the model
    print(f"Using device: {device}")
    print("Loading model...")


    model.eval()

    # Main program loop
    while True:
        choice = display_menu()

        with torch.no_grad():
            if choice == '1':
                print("Starting webcam. Press 'q' to exit...")
                keypoints_on_webcam(model)

            elif choice == '2':
                file_path = select_file()
                if file_path:
                    print(f"Processing image: {file_path}")
                    try:
                        keypoints_on_image(model, file_path)
                    except Exception as e:
                        print(f"Error processing image: {e}")
                else:
                    print("No valid file selected")

            elif choice == '3':
                file_path = select_file()
                if file_path:
                    print(f"Processing video: {file_path}")
                    try:

                        keypoints_on_video(model, file_path)
                    except Exception as e:
                        print(f"Error processing video: {e}")
                else:
                    print("No valid file selected")

            elif choice == '4':
                print("Exiting program...")
                break

            else:
                print("Invalid choice. Please try again.")


if __name__ == '__main__':
    main()