import torch
import time
from dataset import SequenceDataset
from seg_model import RNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "./checkpoints/model_best.pth"
ROOT_DIR = "./filtered_data"

model = RNN(in_channels=1, base_channels=8, num_classes=3).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

dataset = SequenceDataset(ROOT_DIR)

def benchmark_inference_time():
    frame_times = []

    with torch.no_grad():
        for seq_idx in range(len(dataset)):
            sequence = dataset[seq_idx]
            imgs = sequence['images'].to(DEVICE)
            sequence_name = sequence["seq_name"]

            T = imgs.size(0)
            model.h_prev = None

            for t in range(T):
                img_t = imgs[t].unsqueeze(0)
                start_frame_time = time.time()
                
                _ = model(img_t, t)

                torch.cuda.synchronize()

                end_frame_time = time.time()

                frame_times.append(end_frame_time - start_frame_time)

                print(f'{sequence_name} IMG: {t} complete')

        


    avg_frame_time = sum(frame_times) / len(frame_times)
    avg_fps = 1 / avg_frame_time

    print("\n======== INFERENCE SPEED ========")
    print(f"Average frame time: {avg_frame_time*1000:.3f} ms")
    print(f"Average FPS: {avg_fps:.2f}")
    print("=========================================")


if __name__ == "__main__":
    benchmark_inference_time()


