import cv2
import numpy as np
import argparse
import subprocess
import os


def generate_energy_map(frame, flow=None):
    """
    Genera un mapa de energía basado en bordes y flujo óptico (si se proporciona).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    energy_map = np.hypot(sobel_x, sobel_y)

    if flow is not None:
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        energy_map += magnitude

    return energy_map


def find_low_energy_seam(energy_map, axis='vertical'):
    """
    Encuentra la costura de menor energía en el mapa de energía.
    """
    h, w = energy_map.shape
    seam = []

    if axis == 'vertical':
        cost = energy_map.copy()
        backtrack = np.zeros_like(cost, dtype=int)

        for i in range(1, h):
            for j in range(w):
                left = cost[i - 1, j - 1] if j > 0 else np.inf
                up = cost[i - 1, j]
                right = cost[i - 1, j + 1] if j < w - 1 else np.inf

                min_cost = min(left, up, right)
                backtrack[i, j] = np.argmin([left, up, right]) - 1
                cost[i, j] += min_cost

        j = np.argmin(cost[-1])
        for i in range(h - 1, -1, -1):
            seam.append((i, j))
            j += backtrack[i, j]

    else:
        cost = energy_map.T.copy()
        seam = find_low_energy_seam(cost.T, axis='vertical')
        seam = [(j, i) for i, j in seam]

    return seam


def add_seam(frame, seam, axis='vertical'):
    """
    Añade una costura al frame duplicando píxeles en la ruta de menor energía.
    """
    h, w = frame.shape[:2]
    if axis == 'vertical':
        new_frame = np.zeros((h, w + 1, 3), dtype=frame.dtype)
        for i, (row, col) in enumerate(seam):
            new_frame[row, :col] = frame[row, :col]
            new_frame[row, col] = frame[row, col]
            new_frame[row, col + 1:] = frame[row, col:]
    else:
        new_frame = np.zeros((h + 1, w, 3), dtype=frame.dtype)
        for i, (row, col) in enumerate(seam):
            new_frame[:row, col] = frame[:row, col]
            new_frame[row, col] = frame[row, col]
            new_frame[row + 1:, col] = frame[row:, col]

    return new_frame


def update_energy_map(frame, energy_map, seam, axis='vertical'):
    """
    Actualiza el mapa de energía después de añadir una costura.
    """
    h, w = energy_map.shape
    if axis == 'vertical':
        new_energy_map = np.zeros((h, w + 1), dtype=energy_map.dtype)
        for i, (row, col) in enumerate(seam):
            new_energy_map[row, :col] = energy_map[row, :col]
            new_energy_map[row, col] = energy_map[row, col]
            new_energy_map[row, col + 1:] = energy_map[row, col:]
    else:
        new_energy_map = np.zeros((h + 1, w), dtype=energy_map.dtype)
        for i, (row, col) in enumerate(seam):
            new_energy_map[:row, col] = energy_map[:row, col]
            new_energy_map[row, col] = energy_map[row, col]
            new_energy_map[row + 1:, col] = energy_map[row:, col]

    return new_energy_map


def expand_frame_with_seams(frame, target_width, target_height, energy_map):
    """
    Expande un frame agregando costuras basadas en el mapa de energía.
    """
    current_height, current_width = frame.shape[:2]

    while current_width < target_width:
        print(f"Expandiendo ancho: {current_width + 1}/{target_width}")
        seam = find_low_energy_seam(energy_map)
        frame = add_seam(frame, seam)
        energy_map = update_energy_map(frame, energy_map, seam, axis='vertical')
        current_width += 1

    while current_height < target_height:
        print(f"Expandiendo altura: {current_height + 1}/{target_height}")
        seam = find_low_energy_seam(energy_map, axis='horizontal')
        frame = add_seam(frame, seam, axis='horizontal')
        energy_map = update_energy_map(frame, energy_map, seam, axis='horizontal')
        current_height += 1

    return frame


def process_video_custom(input_file, output_file, target_width, target_height, max_frames=None, num_workers=2):
    """
    Procesa un video utilizando seam carving para expandirlo o reducirlo.
    """
    temp_dir = "temp_frames"
    if os.path.exists(temp_dir):
        print("Eliminando carpeta temporal...")
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    if not os.path.exists(input_file):
        print(f"Error: El archivo {input_file} no existe.")
        return

    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"Error: no se pudo abrir el archivo {input_file}")
        return

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    print(f"Procesando video: {input_file}")
    print(f"Resolución original: {original_width}x{original_height}")
    print(f"Resolución objetivo: {target_width}x{target_height}")
    print(f"Total de frames en el video: {frame_count}")

    frame_count = min(frame_count, 10) if max_frames is None else min(frame_count, max_frames)
    print(f"Procesando un máximo de {frame_count} frames")

    cap = cv2.VideoCapture(input_file)
    frame_idx = 0
    prev_frame = None
    processed_frames = []

    while frame_idx < frame_count:
        ret, frame = cap.read()
        if not ret:
            break

        print(f"Procesando frame {frame_idx + 1}/{frame_count}")

        flow = None
        if prev_frame is not None:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        energy_map = generate_energy_map(frame, flow)
        processed_frame = expand_frame_with_seams(frame, target_width, target_height, energy_map)
        processed_frames.append(processed_frame)

        prev_frame = frame
        frame_idx += 1

    cap.release()

    print("Creando video final a partir de los frames procesados...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_file, fourcc, fps, (target_width, target_height))

    for idx, processed_frame in enumerate(processed_frames):
        print(f"Escribiendo frame {idx + 1}/{len(processed_frames)} en el video final")
        out.write(processed_frame)

    out.release()
    print(f"Video procesado y guardado en {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procesar video con seam carving.")
    parser.add_argument("input_file", type=str, help="Ruta del video de entrada.")
    parser.add_argument("output_file", type=str, help="Ruta del video de salida.")
    parser.add_argument("target_width", type=int, help="Ancho objetivo del video.")
    parser.add_argument("target_height", type=int, help="Altura objetivo del video.")
    parser.add_argument("--max_frames", type=int, default=None, help="Número máximo de frames a procesar.")

    args = parser.parse_args()
    process_video_custom(args.input_file, args.output_file, args.target_width, args.target_height, args.max_frames)
