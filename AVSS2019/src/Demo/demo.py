import os
import sys
import cv2
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
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

    frame_index = 0

    # loop through the whole video
    # todo use threading to handle it when having gpus
    while True:
        frames, original_frames = vl.get_frames()

        # vl.get_frames
        if frames is None:
            print("No more frames!")
            break

        x = frames_to_tensor(frames)
        y = be.predict(x)
 
        for i, label in enumerate(y):
            time = (vl.pos - len(y) + i + 1) / fps
            h, m, s = sec_to_hms(time)
            time_str = f"{int(h)}:{int(m)}:{int(s)}"
            all_times.append(time_str)
            print(f"Processing time: {time_str}, Label: {label}")

            if label == 'violent':
                print('violent scene at time:\n%d:%d:%d' % (h, m, s))
                frame = draw_red_border(original_frames[i])
                output_path = os.path.join(output_dir, f'frame_{int(h)}_{int(m)}_{int(s)}_{frame_index:04d}.jpg')
                cv2.imwrite(output_path, frame)
                predicted_labels.append(1)
            else:
                predicted_labels.append(0)
            frame_index += 1
                
    aligned_true_labels = [true_labels_dict.get(time, 0) for time in all_times]

    # Print time, true label, and predicted label
    for time, true_label, pred_label in zip(all_times, aligned_true_labels, predicted_labels):
        print(f"Time: {time}, True Label: {true_label}, Predicted Label: {pred_label}")

    #Calculating confusion matrix and getting TN, FP, FP, TP
    cm = confusion_matrix(aligned_true_labels, predicted_labels)
    TN, FP, FN, TP = cm.ravel()

    #F1 Score formula
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (FN + TP) if (FN + TP) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

    #calculating AUC
    auc = roc_auc_score(aligned_true_labels, predicted_labels)
    print(f'AUC: {auc}')

if __name__ == '__main__':
    main()