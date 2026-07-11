import cv2
import numpy as np
import argparse


def center_crop(frame, crop_size):
    """先把最短邊 resize 到 crop_size（大圖縮小、小圖放大），再 center crop 成 crop_size x crop_size"""
    h, w = frame.shape[:2]

    # 縮放至最短邊 = crop_size
    if min(h, w) != crop_size:
        scale = crop_size / min(h, w)
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        new_w = max(crop_size, int(round(w * scale)))
        new_h = max(crop_size, int(round(h * scale)))
        frame = cv2.resize(frame, (new_w, new_h), interpolation=interp)
        h, w = frame.shape[:2]

    # center crop
    start_x = (w - crop_size) // 2
    start_y = (h - crop_size) // 2
    return frame[start_y:start_y + crop_size, start_x:start_x + crop_size]


def create_diagonal_mask_with_line(width, height, position, line_width=9):
    """
    創建傾斜分割線遮罩（支援任意長寬）
    position: 0-1 之間的值，表示分割線的位置
    0 = 完全顯示 input
    1 = 完全顯示 output
    0.5 = 分割線在中間（上方在右邊 1/3 處，下方在左邊 1/3 處）

    返回:
    - mask: 255 = 顯示 input, 0 = 顯示 output
    - line_mask: 分割線的位置
    """
    y_coords, x_coords = np.mgrid[0:height, 0:width]

    # 定義線的方程式 f(x,y) = 3x/W + y/H
    # f 的範圍: 0 (左上角) 到 4 (右下角)
    f = 3.0 * x_coords / width + y_coords.astype(float) / height

    # position 0->1 映射到 threshold 4->0
    max_f = 4.0
    threshold = max_f * (1 - position)

    # f < threshold 的區域顯示 input (255)
    mask = (f < threshold).astype(np.uint8) * 255

    # 計算點到線的像素距離
    gradient_magnitude = np.sqrt((3.0 / width) ** 2 + (1.0 / height) ** 2)
    pixel_distance = np.abs(f - threshold) / gradient_magnitude

    # line_mask: 分割線附近的區域
    line_mask = (pixel_distance <= line_width / 2).astype(np.uint8) * 255

    return mask, line_mask


def apply_diagonal_line(image, line_mask, line_color=(255, 255, 255)):
    """在圖片上應用分割線"""
    line_mask_3channel = cv2.cvtColor(line_mask, cv2.COLOR_GRAY2BGR) / 255.0
    line_layer = np.full_like(image, line_color, dtype=np.uint8)
    result = (image * (1 - line_mask_3channel) + line_layer * line_mask_3channel).astype(np.uint8)
    return result


def create_transition_frame(input_frame, output_frame, position, line_width=9):
    """
    Create a single transition frame.
    position: 0-1, 分割線位置
    """
    h, w = input_frame.shape[:2]

    # Create diagonal mask
    mask, line_mask = create_diagonal_mask_with_line(w, h, position, line_width)

    # Expand mask to 3 channels
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0

    # Input -> Output: mask=255 顯示 input, mask=0 顯示 output
    blended = (input_frame * mask_3ch + output_frame * (1 - mask_3ch)).astype(np.uint8)

    # Apply white gap line
    blended = apply_diagonal_line(blended, line_mask, line_color=(255, 255, 255))

    return blended


def create_dummy_frame(width, height, text, color):
    """Create a dummy colored frame with text for testing"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = color

    # Add text
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.0
    thickness = 2
    text_color = (255, 255, 255)

    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (width - text_width) // 2
    y = (height + text_height) // 2

    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness)

    return frame


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate an input/output comparison video with a diagonal wipe transition')
    parser.add_argument('--input', default='input.mp4',
                        help='input video path (default: input.mp4)')
    parser.add_argument('--output', default='output.mp4',
                        help='output video path (default: output.mp4)')
    parser.add_argument('--result', default='transition_result.mp4',
                        help='result video path (default: transition_result.mp4)')
    parser.add_argument('--crop', type=int, default=384, metavar='SIZE',
                        help='center crop size in pixels (default: 384)')
    parser.add_argument('--no-crop', action='store_true',
                        help='disable cropping and use the original input video size')
    parser.add_argument('--fps', type=float, default=None,
                        help='output video fps (default: same as input video)')
    parser.add_argument('--resize', type=int, default=None, metavar='SHORT_SIDE',
                        help='resize the result so its shortest side equals SHORT_SIDE, '
                             'keeping aspect ratio (default: no resize)')
    parser.add_argument('--transition-frames', type=int, default=31,
                        help='number of frames in the transition animation (default: 31)')
    parser.add_argument('--line-width', type=int, default=9,
                        help='white split line width in pixels (default: 9)')
    return parser.parse_args()


def main():
    args = parse_args()
    crop_size = None if args.no_crop else args.crop
    if crop_size is not None and crop_size <= 0:
        print("Error: --crop must be a positive integer (or use --no-crop)")
        return
    if args.fps is not None and args.fps <= 0:
        print("Error: --fps must be positive")
        return
    if args.resize is not None and args.resize <= 0:
        print("Error: --resize must be positive")
        return

    # Load videos
    print("Loading videos...")
    input_cap = cv2.VideoCapture(args.input)
    output_cap = cv2.VideoCapture(args.output)

    if not input_cap.isOpened():
        print(f"Error: Could not open input video {args.input}")
        return
    if not output_cap.isOpened():
        print(f"Error: Could not open output video {args.output}")
        return

    # 自動取得影片資訊
    input_fps = input_cap.get(cv2.CAP_PROP_FPS)
    output_fps = output_cap.get(cv2.CAP_PROP_FPS)
    input_total = int(input_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_total = int(output_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = min(input_total, output_total)  # 取兩者最小值

    # result fps：優先使用 --fps，否則沿用 input 影片的 fps
    result_fps = args.fps if args.fps is not None else input_fps

    # 決定基準 frame 大小
    if crop_size is not None:
        frame_w = frame_h = crop_size
        size_note = f"{frame_w}x{frame_h} (center crop)"
    else:
        frame_w = int(input_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(input_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        size_note = f"{frame_w}x{frame_h} (original size)"

    # --resize：等比縮放到短邊 = SHORT_SIDE（長邊取偶數以利編碼相容性）
    if args.resize is not None and args.resize != min(frame_w, frame_h):
        scale = args.resize / min(frame_w, frame_h)
        if frame_w <= frame_h:
            frame_w, frame_h = args.resize, max(2, int(round(frame_h * scale / 2)) * 2)
        else:
            frame_w, frame_h = max(2, int(round(frame_w * scale / 2)) * 2), args.resize
        size_note += f", resized to {frame_w}x{frame_h}"

    # no-crop 時若 output 影片尺寸和最終大小不同，會自動 resize 對齊
    if crop_size is None:
        out_w = int(output_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        out_h = int(output_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (out_w, out_h) != (frame_w, frame_h):
            print(f"  Warning: output video is {out_w}x{out_h}, "
                  f"frames will be resized to {frame_w}x{frame_h}")

    print(f"  Input video: {input_total} frames at {input_fps:.2f} fps")
    print(f"  Output video: {output_total} frames at {output_fps:.2f} fps")
    print(f"  Using: {total_frames} frames")
    fps_note = "user specified" if args.fps is not None else "same as input"
    print(f"  Result fps: {result_fps:.2f} ({fps_note})")
    print(f"  Frame size: {size_note}")

    def process_frame(frame):
        """依設定做 center crop，再 resize 成最終輸出大小"""
        if crop_size is not None:
            frame = center_crop(frame, crop_size)
        if frame.shape[1] != frame_w or frame.shape[0] != frame_h:
            interp = cv2.INTER_AREA if frame.shape[1] > frame_w else cv2.INTER_LINEAR
            frame = cv2.resize(frame, (frame_w, frame_h), interpolation=interp)
        return frame

    # 計算轉換的起始和結束幀（以中間為中心）
    transition_frames = args.transition_frames
    center_frame = total_frames // 2
    transition_start = center_frame - transition_frames // 2
    transition_end = center_frame + transition_frames // 2

    # Setup output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.result, fourcc, result_fps, (frame_w, frame_h))

    print("Generating transition video...")
    print(f"  Transition: frame {transition_start} ~ {transition_end}")

    # 保留最後有效幀
    last_input_frame = None
    last_output_frame = None

    # 逐幀處理
    for frame_idx in range(total_frames):
        # 同步讀取兩個影片的當前幀
        ret1, input_frame = input_cap.read()
        ret2, output_frame = output_cap.read()

        # 如果讀取成功，更新並保留；失敗則使用最後有效幀
        if ret1 and input_frame is not None:
            input_frame = process_frame(input_frame)
            last_input_frame = input_frame
        else:
            input_frame = last_input_frame

        if ret2 and output_frame is not None:
            output_frame = process_frame(output_frame)
            last_output_frame = output_frame
        else:
            output_frame = last_output_frame

        # 確保有有效幀
        if input_frame is None:
            input_frame = create_dummy_frame(frame_w, frame_h, "Input", (50, 50, 150))
        if output_frame is None:
            output_frame = create_dummy_frame(frame_w, frame_h, "Output", (50, 150, 50))

        # 決定輸出內容
        if frame_idx < transition_start:
            # Phase 1: 顯示 input
            out.write(input_frame)
        elif frame_idx < transition_end:
            # Phase 2: 轉換動畫
            t = frame_idx - transition_start
            position = t / max(transition_frames - 1, 1)
            frame = create_transition_frame(input_frame, output_frame, position,
                                            args.line_width)
            out.write(frame)
        else:
            # Phase 3: 顯示 output
            out.write(output_frame)

    # Cleanup
    out.release()
    input_cap.release()
    output_cap.release()

    print(f"Done! Generated {total_frames} frames.")
    print(f"Output saved to: {args.result}")


if __name__ == "__main__":
    main()
