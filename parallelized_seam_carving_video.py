import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import os
from glob import glob
import argparse
import logging
import sys
import maxflow

logging.basicConfig(
    filename="seam_carving.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",
    level=logging.INFO,
)


def usage():
    print(
        "Usage:python3 parallelized_seam_carving_video.py <filename> <desired height> <desired width> <num workers> <output file>"
    )


def remove_seam(image, seam):

    nrows, ncols, channels = image.shape
    # Initialize a reduced image with one less column
    reduced_image = np.zeros((nrows, ncols - 1, channels), dtype=image.dtype)

    for i in range(nrows):

        # If seam is not at the first column, copy pixels before the seam
        if seam[i] != 0:
            reduced_image[i, : seam[i]] = image[i, : seam[i]]
        # If seam is not at the last column, copy pixels after the seam
        if seam[i] < ncols - 1:
            reduced_image[i, seam[i] :] = image[i, seam[i] + 1 :]

    return reduced_image


def remove_seam_gray(gray_image, seam):
    nrows, ncols = gray_image.shape
    # Initialize a reduced image with one less column
    reduced_image = np.zeros((nrows, ncols - 1), dtype=gray_image.dtype)

    for i in range(nrows):
        # If seam is not at the first column, copy pixels before the seam
        if seam[i] > 0:
            reduced_image[i, : seam[i]] = gray_image[i, : seam[i]]
        # If seam is not at the last column, copy pixels after the seam
        if seam[i] < ncols - 1:
            reduced_image[i, seam[i] :] = gray_image[i, seam[i] + 1 :]

    return reduced_image


def find_seam(grayImg1, grayImg2, grayImg3, grayImg4, grayImg5):

    rows, cols = grayImg1.shape
    inf = 100000
    seam = np.zeros(rows, dtype=int)
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    # weights = np.array([0.4, 0.2, 0.15, 0.15, 0.1])
    # weights = np.array([0.3, 0.1, 0.2, 0.1, 0.3])
    grayImgs = np.stack(
        [
            grayImg1.astype(np.uint8),
            grayImg2.astype(np.uint8),
            grayImg3.astype(np.uint8),
            grayImg4.astype(np.uint8),
            grayImg5.astype(np.uint8),
        ],
        axis=-1,
    )

    # Create the graph
    g = maxflow.Graph[int](
        rows * cols, (rows - 1) * cols + (cols - 1) * rows + 2 * (rows - 1) * (cols - 1)
    )
    nodes = g.add_nodes(rows * cols)

    # Add edges to the graph
    for i in range(rows):
        for j in range(cols):
            node_id = i * cols + j

            # Add terminal edges
            if j == 0:
                g.add_tedge(node_id, inf, 0)
            elif j == cols - 1:
                g.add_tedge(node_id, 0, inf)

            # Add intra-row edges (left-to-right)
            if j < cols - 1:
                next_col_values = grayImgs[i, j + 1]
                if j == 0:
                    LR = np.sum(next_col_values * weights)
                else:
                    prev_col_values = grayImgs[i, j - 1]
                    LR = np.sum(np.abs(next_col_values - prev_col_values) * weights)
                g.add_edge(node_id, node_id + 1, int(LR), inf)

            # Add inter-row edges (top-to-bottom)
            if i < rows - 1:
                if j == 0:
                    posLU = np.sum(grayImgs[i, j] * weights)
                    negLU = np.sum(grayImgs[i + 1, j] * weights)
                else:
                    LU_curr = grayImgs[i, j]
                    LU_prev_col = grayImgs[i + 1, j - 1]
                    posLU = np.sum(np.abs(LU_curr - LU_prev_col) * weights)

                    LU_next_row = grayImgs[i + 1, j]
                    LU_prev_curr = grayImgs[i, j - 1]
                    negLU = np.sum(np.abs(LU_next_row - LU_prev_curr) * weights)

                g.add_edge(node_id, node_id + cols, int(negLU), int(posLU))

            # Add diagonal edges
            if i != 0 and j != 0:
                g.add_edge(node_id, (i - 1) * cols + j - 1, inf, 0)
            if i != rows - 1 and j != 0:
                g.add_edge(node_id, (i + 1) * cols + j - 1, inf, 0)

    # Perform max flow
    g.maxflow()

    # Extract seam
    segments = np.array([g.get_segment(i) for i in range(rows * cols)])
    segments = segments.reshape(rows, cols)

    for i in range(rows):
        sink_nodes = np.where(segments[i] == 1)[0]
        if sink_nodes.size > 0:
            seam[i] = (
                sink_nodes[0] - 1 if sink_nodes[0] > 0 else 0
            )  # Adjust indexing to match C++ logic
        else:
            seam[i] = cols - 1  # Fallback for entire row in source

    return seam


def reduce_vertical(gray_img1, gray_img2, gray_img3, gray_img4, gray_img5, img):
    seam = find_seam(gray_img1, gray_img2, gray_img3, gray_img4, gray_img5)
    img = remove_seam(img, seam)
    gray_img1 = remove_seam_gray(gray_img1, seam)
    gray_img2 = remove_seam_gray(gray_img2, seam)
    gray_img3 = remove_seam_gray(gray_img3, seam)
    gray_img4 = remove_seam_gray(gray_img4, seam)
    gray_img5 = remove_seam_gray(gray_img5, seam)
    return img, gray_img1, gray_img2, gray_img3, gray_img4, gray_img5


def reduce_horizontal(gray_img1, gray_img2, gray_img3, gray_img4, gray_img5, img):
    # Find seam on transposed images
    seam = find_seam(gray_img1.T, gray_img2.T, gray_img3.T, gray_img4.T, gray_img5.T)

    # Remove seam from all images and return the updated images
    img = remove_seam(img.transpose(1, 0, 2), seam).transpose(1, 0, 2)
    gray_img1 = remove_seam_gray(gray_img1.T, seam).T
    gray_img2 = remove_seam_gray(gray_img2.T, seam).T
    gray_img3 = remove_seam_gray(gray_img3.T, seam).T
    gray_img4 = remove_seam_gray(gray_img4.T, seam).T
    gray_img5 = remove_seam_gray(gray_img5.T, seam).T
    return img, gray_img1, gray_img2, gray_img3, gray_img4, gray_img5


def reduce_frame(frame1, frame2, frame3, frame4, frame5, v, h):
    img = frame1.copy()
    gray_img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray_img3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
    gray_img4 = cv2.cvtColor(frame4, cv2.COLOR_BGR2GRAY)
    gray_img5 = cv2.cvtColor(frame5, cv2.COLOR_BGR2GRAY)

    for _ in range(v):
        img, gray_img1, gray_img2, gray_img3, gray_img4, gray_img5 = reduce_vertical(
            gray_img1, gray_img2, gray_img3, gray_img4, gray_img5, img
        )

    for _ in range(h):
        img, gray_img1, gray_img2, gray_img3, gray_img4, gray_img5 = reduce_horizontal(
            gray_img1, gray_img2, gray_img3, gray_img4, gray_img5, img
        )
    return img


def create_video_from_images(image_dir, output_file, fps=30):
    # Get all images in the directory, sorted by file name
    images = sorted(glob(os.path.join(image_dir, "*.png")))

    if not images:
        logging.error("No images found in the directory.")
        return

    # Read the first image to get dimensions
    first_image = cv2.imread(images[0])
    height, width, layers = first_image.shape

    # Initialize VideoWriter for AVI
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Codec for AVI
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for img_path in images:
        img = cv2.imread(img_path)
        video.write(img)

    video.release()
    logging.info(f"Video saved as {output_file}")


def process_frame(args):
    frames, frame_id, max_frames, v, h = args
    frame1 = frames[frame_id]
    frame2 = frames[frame_id + 1] if frame_id + 1 < max_frames else frame1
    frame3 = frames[frame_id + 2] if frame_id + 2 < max_frames else frame2
    frame4 = frames[frame_id + 3] if frame_id + 3 < max_frames else frame3
    frame5 = frames[frame_id + 4] if frame_id + 4 < max_frames else frame4

    # print(f"Processing frame {frame_id + 1}/{max_frames}...")
    logging.info(f"Processing frame {frame_id + 1}/{max_frames}...")
    return reduce_frame(frame1, frame2, frame3, frame4, frame5, v, h)


def initialize_video_capture(in_file):
    cap = cv2.VideoCapture(in_file)
    if not cap.isOpened():
        logging.error("Unable to open input file.")
        return None, None, None, None, None

    max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    return cap, max_frames, orig_width, orig_height, fps


def extract_frames(cap, max_frames):
    frames = []
    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        logging.error("No frames extracted from video.")
        return None
    return frames


def process_video(frames, max_frames, orig_width, orig_height, ver, hor, num_workers):

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        args = [(frames, i, max_frames, ver, hor) for i in range(max_frames)]
        out_frames = list(executor.map(process_frame, args))

    return out_frames


def save_output_video(out_frames, out_file, fps, orig_width, orig_height, ver, hor):
    output = cv2.VideoWriter(
        out_file,
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (orig_width - ver, orig_height - hor),
    )

    for out_frame in out_frames:
        output.write(out_frame)

    output.release()
    logging.info(f"Output file saved as: {out_file}")


def main(args):
    cap, max_frames, orig_width, orig_height, fps = initialize_video_capture(
        args.filename
    )
    if not cap:
        sys.exit(-1)

    frames = extract_frames(cap, max_frames)
    if not frames:
        sys.exit(-1)

    ver = abs(orig_width - args.desired_width)
    hor = abs(orig_height - args.desired_height)
    logging.info(f"Original dimensions: {orig_width}x{orig_height}, FPS: {fps}")
    logging.info(f"Resizing to: {args.desired_width}x{args.desired_height}")

    out_frames = process_video(
        frames, max_frames, orig_width, orig_height, ver, hor, args.num_workers
    )
    # out_file = in_file.split(".")[0] + "_result.avi"
    save_output_video(
        out_frames,
        args.output_file,
        fps,
        orig_width,
        orig_height,
        ver,
        hor,
    )
    logging.info("Video processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reduce video dimensions using seam carving based on sequential frames."
    )
    parser.add_argument(
        "-f", "--filename", type=str, help="Path to the input video file."
    )
    parser.add_argument(
        "-dh", "--desired_height", type=int, help="Desired height of the output video."
    )
    parser.add_argument(
        "-dw", "--desired_width", type=int, help="Desired width of the output video."
    )
    parser.add_argument(
        "-nw",
        "--num_workers",
        type=int,
        help="Number of workers for parallel processing.",
    )
    parser.add_argument(
        "-o", "--output_file", type=str, help="Path to the output video file."
    )

    args = parser.parse_args()
    main(args)
