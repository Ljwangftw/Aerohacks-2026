# Quick test script to check whether both cameras are detected and working
import cv2


def main():
    # Open the first and second camera devices
    cap0 = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(1)

    # Stop if camera 0 could not be opened
    if not cap0.isOpened():
        print("Could not open camera 0")
        return

    # Stop if camera 1 could not be opened
    if not cap1.isOpened():
        print("Could not open camera 1")
        return

    # Keep reading and showing frames until the user quits
    while True:
        # Read one frame from each camera
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        # Stop if camera 0 fails to provide a frame
        if not ret0 or frame0 is None:
            print("Failed to read from camera 0")
            break

        # Stop if camera 1 fails to provide a frame
        if not ret1 or frame1 is None:
            print("Failed to read from camera 1")
            break

        # Show the live video feed from both cameras
        cv2.imshow("Camera 0", frame0)
        cv2.imshow("Camera 1", frame1)

        # Check for keyboard input every loop iteration
        key = cv2.waitKey(1) & 0xFF

        # Press q to exit the program
        if key == ord("q"):
            break

    # Release both cameras so they are no longer in use
    cap0.release()
    cap1.release()

    # Close all OpenCV display windows
    cv2.destroyAllWindows()


# Run main() only when this file is executed directly
if __name__ == "__main__":
    main()