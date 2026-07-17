import os, sys, cv2
import numpy as np

def dota_to_labels(datapath):
    classes = ['airplane', 'ship', 'storage-tank', 'baseball-diamond',
               'tennis-court', 'basketball-court', 'ground-track-field',
               'harbor', 'bridge', 'vehicle']

    for subset in ['training', 'evaluation']:
        lbl_dir = os.path.join(datapath, subset, 'labels')
        os.makedirs(lbl_dir, exist_ok=True)
        img_dir = os.path.join(datapath, subset, 'images')
        src_dir = os.path.join(datapath, subset, 'labelTxt')
        if not os.path.exists(src_dir):
            print(f"SKIP: {src_dir} not found")
            continue
        for fn in os.listdir(img_dir):
            if not fn.lower().endswith(('.jpg', '.png')):
                continue
            img = cv2.imread(os.path.join(img_dir, fn))
            if img is None:
                continue
            base = os.path.splitext(fn)[0]
            src = os.path.join(src_dir, base + '.txt')
            dst = os.path.join(lbl_dir, base + '.txt')
            if not os.path.exists(src):
                continue
            lines = []
            with open(src, 'r') as f:
                for line in f:
                    p = line.strip().split()
                    if len(p) < 10 or p[-1] == '2':
                        continue
                    if p[8] not in classes:
                        continue
                    cid = classes.index(p[8])
                    pts = np.array([float(x) for x in p[:8]], dtype=np.float32).reshape(4, 2)
                    (cx, cy), (bw, bh), _ = cv2.minAreaRect(pts)
                    lines.append(f'{cid} {cx:.2f} {cy:.2f} {bw:.2f} {bh:.2f}\n')
            with open(dst, 'w') as f:
                f.writelines(lines)

if __name__ == '__main__':
    dota_to_labels(os.path.abspath(sys.argv[1]))
