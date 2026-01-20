import os
import cv2
import argparse
import numpy as np
import random
from math import cos, sin, pi

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def blank_image(size):
    return np.zeros((size, size), dtype=np.uint8)

def random_center(size):
    margin = size // 5
    return random.randint(margin, size - margin), random.randint(margin, size - margin)

def draw_circle(img, mask):
    center = random_center(img.shape[0])
    radius = random.randint(15, img.shape[0] // 6)
    cv2.circle(mask, center, radius, 255, -1)
    cv2.circle(img, center, radius, 255, 1)

def draw_rectangle(img, mask):
    size = img.shape[0]
    w, h = random.randint(30, size//4), random.randint(30, size//4)
    cx, cy = random_center(size)
    rect = np.array([[cx - w//2, cy - h//2],
                     [cx + w//2, cy - h//2],
                     [cx + w//2, cy + h//2],
                     [cx - w//2, cy + h//2]])
    cv2.fillPoly(mask, [rect], 255)
    cv2.polylines(img, [rect], True, 255, 1)

def draw_ellipse(img, mask):
    center = random_center(img.shape[0])
    axes = (random.randint(20, 80), random.randint(20, 80))
    angle = random.randint(0, 180)
    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
    cv2.ellipse(img, center, axes, angle, 0, 360, 255, 1)

def draw_star(img, mask):
    cx, cy = random_center(img.shape[0])
    r1, r2 = random.randint(20,40), 0
    r2 = r1 // 2
    points = []
    for i in range(10):
        angle = i * pi / 5
        r = r1 if i % 2 == 0 else r2
        x = int(cx + r * cos(angle))
        y = int(cy + r * sin(angle))
        points.append([x, y])
    pts = np.array(points)
    cv2.fillPoly(mask, [pts], 255)
    cv2.polylines(img, [pts], True, 255, 1)

def draw_dot(img, mask):
    center = random_center(img.shape[0])
    cv2.circle(mask, center, 3, 255, -1)
    cv2.circle(img, center, 3, 255, -1)

def draw_line(img, mask):
    size = img.shape[0]
    x1, y1 = random_center(size)
    x2, y2 = random_center(size)
    cv2.line(img, (x1, y1), (x2, y2), 255, 1)
    cv2.line(mask, (x1, y1), (x2, y2), 255, 1)

SHAPES = [draw_circle, draw_rectangle, draw_ellipse, draw_star, draw_dot, draw_line]

def main(args):
    out_outline = os.path.join(args.output, args.split, "outlines")
    out_filled = os.path.join(args.output, args.split, "filled")
    ensure_dir(out_outline)
    ensure_dir(out_filled)

    for i in range(args.count):
        img = blank_image(args.size)
        mask = blank_image(args.size)
        random.choice(SHAPES)(img, mask)
        cv2.imwrite(f"{out_outline}/{i:05d}.png", img)
        cv2.imwrite(f"{out_filled}/{i:05d}.png", mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="../data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    main(args)
