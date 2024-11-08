import os
import sys
import cv2
import json
import datetime
from sklearn.metrics import confusion_matrix, f1_score
from utils import sec_to_hms
from backend import Backend
from videoloader import VideoLoader, frames_to_tensor, draw_red_border

def load_true_labels(label_file):
    true_labels = []
    with open(label_file, 'r') as file:
        for line in file:
            time_str, label = line.strip().split(',')
            true_labels.append((time_str, int(label)))
    return true_labels

def main():
    if len(sys.argv) < 3:
        print("Usage: python demo.py <video_path> <label_file>")
        return

    video_path = sys.argv[1]
    label_file = sys.argv[2]

    if not os.path.exists(video_path):
        print(f"Video file {video_path} does not exist.")
        return

    if not os.path.exists(label_file):
        print(f"Label file {label_file} does not exist.")
        return

    be = Backend()
    vl = VideoLoader(video_path)
    fps = vl.fps

    true_labels = load_true_labels(label_file)
    true_labels_dict = {time: label for time, label in true_labels}
    predicted_labels = []
    all_times = []

    output_dir = 'output_frames'
    os.makedirs(output_dir, exist_ok=True)

    json_output = {"fps": fps, "frames": []}
    frame_index = 0

    while True:
        frames, original_frames = vl.get_frames()
        if frames is None:
            print("No more frames!")
            break

        x = frames_to_tensor(frames)
        y = be.predict(x)

        for i, pred_label in enumerate(y):
            time = (vl.pos - len(y) + i + 1) / fps
            h, m, s = sec_to_hms(time)
            duration_str = f"{int(h)}:{int(m)}:{int(s)}"
            current_date = datetime.datetime.now().strftime("%m/%d/%Y")
            full_timestamp = f"{current_date} {int(h):02}:{int(m):02}:{int(s):02}-{frame_index:03d}"
            all_times.append(f"{int(h)}:{int(m)}:{int(s)}")

            # Get the ground truth label
            ground_truth_label = true_labels_dict.get(f"{int(h)}:{int(m)}:{int(s)}", 0)
            label_text = "violent" if ground_truth_label == 1 else "normal"

            # Print according to the specified format
            print(f"Processing time: {duration_str}, Label: {label_text}")

            # If the ground truth label is violent, add an additional line
            if ground_truth_label == 1:
                print(f"Violent scene at {duration_str}")
                frame = draw_red_border(original_frames[i])

                # Overlay text for violent frames
                text = "Violent Scene Detected"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_color = (0, 0, 255)
                thickness = 2
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = (frame.shape[1] - text_size[0]) // 2
                text_y = 50
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, thickness)

                output_path = os.path.join(output_dir, f'frame_{int(h)}_{int(m)}_{int(s)}_{frame_index:04d}.jpg')
                cv2.imwrite(output_path, frame)
                predicted_labels.append(1)
            else:
                frame = original_frames[i]
                predicted_labels.append(0)

            # Display the frame
            cv2.imshow('Violence Detection', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            # Add frame details to JSON output
            json_output["frames"].append({
                "frame": frame_index,
                "timestamp": full_timestamp,
                "label": label_text,
            })

            frame_index += 1

    cv2.destroyAllWindows()

    aligned_true_labels = [true_labels_dict.get(time, 0) for time in all_times]

    if not aligned_true_labels or not predicted_labels:
        print("Error: No valid data for calculation.")
        return

    cm = confusion_matrix(aligned_true_labels, predicted_labels)
    TN, FP, FN, TP = cm.ravel()
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (FN + TP) if (FN + TP) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

    json_path = os.path.join(output_dir, 'output.json')
    with open(json_path, 'w') as json_file:
        json.dump(json_output, json_file, indent=4)

if __name__ == '__main__':
    main()
