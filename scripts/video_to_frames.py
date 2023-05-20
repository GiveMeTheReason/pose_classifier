import os

import cv2


def main():
    source_filename = 'color.avi'
    save_folder = 'test_frames_data'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)

    cap = cv2.VideoCapture(source_filename)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        save_path = os.path.join(
            save_folder,
            f'frame_{str(i).zfill(3)}.jpg',
        )
        cv2.imwrite(save_path, frame)
        i += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
