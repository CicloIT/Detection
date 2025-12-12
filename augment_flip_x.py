import os
import argparse
from PIL import Image


def flip_vertical_image(src_path, dst_path):
    img = Image.open(src_path)
    flipped = img.transpose(Image.FLIP_TOP_BOTTOM)
    flipped.save(dst_path)


def flip_y_yolo(y):
    # YOLO normalized y center becomes 1 - y when flipping over X axis
    return 1.0 - y


def process(images_dir, labels_dir, suffix, mode, sample_ratio):
    imgs = [f for f in sorted(os.listdir(images_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total = len(imgs)
    if total == 0:
        print('No se encontraron imágenes en', images_dir)
        return

    if mode == 'half':
        imgs = imgs[: max(1, total // 2)]
    elif mode == 'sample':
        k = max(1, int(total * float(sample_ratio)))
        imgs = imgs[:k]

    print(f'Procesando {len(imgs)} de {total} imágenes (modo={mode})')

    for i, img_name in enumerate(imgs, 1):
        src_img = os.path.join(images_dir, img_name)
        name, ext = os.path.splitext(img_name)
        dst_img_name = f"{name}{suffix}{ext}"
        dst_img = os.path.join(images_dir, dst_img_name)

        if os.path.exists(dst_img):
            print(f'[{i}] Ya existe {dst_img_name}, salteando')
            continue

        try:
            flip_vertical_image(src_img, dst_img)
        except Exception as e:
            print(f'[{i}] Error al voltear {img_name}:', e)
            continue

        # process label
        src_label = os.path.join(labels_dir, f"{name}.txt")
        dst_label = os.path.join(labels_dir, f"{name}{suffix}.txt")

        if os.path.exists(src_label):
            try:
                with open(src_label, 'r', encoding='utf-8') as f:
                    lines = [l.strip() for l in f if l.strip()]

                out_lines = []
                for ln in lines:
                    parts = ln.split()
                    if len(parts) < 5:
                        # keep line as-is if unexpected format
                        out_lines.append(ln)
                        continue
                    cls = parts[0]
                    x = float(parts[1])
                    y = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    y2 = flip_y_yolo(y)
                    out_lines.append(f"{cls} {x:.6f} {y2:.6f} {w:.6f} {h:.6f}")

                with open(dst_label, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(out_lines) + ('\n' if out_lines else ''))
            except Exception as e:
                print(f'[{i}] Error al procesar label {src_label}:', e)
        else:
            # no label for this image
            pass

        if i % 50 == 0:
            print(f'  Procesadas {i}/{len(imgs)}')


def main():
    p = argparse.ArgumentParser(description='Flip images vertical and create YOLOv8 labels')
    p.add_argument('--images', default='train/images', help='Carpeta de imágenes')
    p.add_argument('--labels', default='train/labels', help='Carpeta de labels')
    p.add_argument('--suffix', default='_flipX', help='Sufijo para imágenes y labels creados')
    p.add_argument('--mode', choices=['all', 'half', 'sample'], default='half', help='Procesar todas, la mitad o una muestra')
    p.add_argument('--sample-ratio', default=0.5, help='Si mode=sample, fracción a procesar (0-1)')
    args = p.parse_args()

    images_dir = os.path.abspath(args.images)
    labels_dir = os.path.abspath(args.labels)

    if not os.path.isdir(images_dir):
        print('No existe la carpeta de imágenes:', images_dir)
        return
    if not os.path.isdir(labels_dir):
        print('No existe la carpeta de labels:', labels_dir)
        return

    process(images_dir, labels_dir, args.suffix, args.mode, args.sample_ratio)


if __name__ == '__main__':
    main()
