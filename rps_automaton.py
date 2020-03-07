import numpy as np
from scipy.signal import convolve2d
import cv2
from tqdm import tqdm
import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument(
    '--width', type=int,
    default=160, help='Width of grid.'
)
parser.add_argument(
    '--height', type=int,
    default=90, help='Height of grid.'
)
parser.add_argument(
    '--num-colours', type=int,
    default=3, help='Number of states in the automaton.'
)
parser.add_argument(
    '--neighbour-threshold', type=int,
    default=3, help='Number of neighbours needed for a transition.'
)
parser.add_argument(
    '--seconds', type=int,
    default=20, help='Number of seconds (at 15 FPS).'
)
args = parser.parse_args()

initial_seconds = 1
fps = 15

colours = [
    np.array([[31,119,180]]),
    np.array([[255,127,14]]),
    np.array([[44,160,44]]),
    np.array([[214,39,40]]),
    np.array([[148,103,189]]),
    np.array([[140,86,75]]),
    np.array([[227,119,194]]),
    np.array([[127, 127, 127]]),
    np.array([[188, 189, 34]]),
    np.array([[23, 190, 207]])
]

convolution = np.array(
    [[1, 1, 1],
     [1, 0, 1],
     [1, 1, 1]]
)


def update(grid):
    new_grid = np.copy(grid)

    colour_grids = [grid == i for i in range(args.num_colours)]

    ns = np.arange(args.num_colours)
    for i, j in zip(ns, np.roll(ns, 1)):
        target_mask = colour_grids[i]
        neighbour_grid = colour_grids[j]
        neighbour_mask = convolve2d(
            neighbour_grid,
            convolution,
            mode='same',
            boundary='wrap'
        ) >= args.neighbour_threshold

        mask = np.logical_and(
            target_mask,
            neighbour_mask
        )
        new_grid[mask] = j
    return new_grid


def make_image(frame_i, grid, image):
    for i in range(args.num_colours):
        mask = grid == i
        image[mask] = colours[i]

    resize_factor = 8
    out_image = cv2.resize(
        image,
        (resize_factor * args.width, resize_factor * args.height),
        interpolation=cv2.INTER_NEAREST
    )
    cv2.imwrite(f'frames/{frame_i:04d}.png', out_image)


if __name__ == '__main__':
    Path('frames').mkdir(exist_ok=True)

    grid = np.random.choice(args.num_colours, size=(args.height, args.width))
    image = np.zeros((args.height, args.width, 3), dtype=np.uint8)

    initial_frames = initial_seconds * fps
    for frame_i in tqdm(range(initial_frames)):
        make_image(frame_i, grid, image)

    subsequent_frames = args.seconds * fps
    for frame_i in tqdm(range(initial_frames, initial_frames + subsequent_frames)):
        make_image(frame_i, grid, image)
        grid = update(grid)

