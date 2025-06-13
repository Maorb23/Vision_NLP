#!/usr/bin/env python3

"""
Split video into scenes using PySceneDetect.

This script provides a command-line interface for splitting videos into scenes using various detection algorithms.
It supports multiple detection methods, preview image generation, and customizable parameters for fine-tuning
the scene detection process.

Basic usage:
    # Split video using default content-based detection
    scenes_split.py input.mp4 output_dir/

    # Save 3 preview images per scene
    scenes_split.py input.mp4 output_dir/ --save-images 3

    # Process specific duration and filter short scenes
    scenes_split.py input.mp4 output_dir/ --duration 60s --filter-shorter-than 2s

Advanced usage:
    # Content detection with minimum scene length and frame skip
    scenes_split.py input.mp4 output_dir/ --detector content --min-scene-length 30 --frame-skip 2

    # Use adaptive detection with custom detector and detector parameters
    scenes_split.py input.mp4 output_dir/ --detector adaptive --threshold 3.0 --adaptive-window 10
"""

import argparse
import os
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

from scenedetect import (
    AdaptiveDetector,
    ContentDetector,
    HistogramDetector,
    SceneManager,
    ThresholdDetector,
    open_video,
)
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.scene_manager import SceneDetector, write_scene_list_html
from scenedetect.scene_manager import save_images as save_scene_images
from scenedetect.stats_manager import StatsManager
from scenedetect.video_splitter import split_video_ffmpeg


class DetectorType(str, Enum):
    """Available scene detection algorithms."""

    CONTENT = "content"  # Detects fast cuts using HSV color space
    ADAPTIVE = "adaptive"  # Detects fast two-phase cuts
    THRESHOLD = "threshold"  # Detects fast cuts/slow fades in from and out to a given threshold level
    HISTOGRAM = "histogram"  # Detects based on YUV histogram differences in adjacent frames


def create_detector(
    detector_type: DetectorType,
    threshold: Optional[float] = None,
    min_scene_len: Optional[int] = None,
    luma_only: Optional[bool] = None,
    adaptive_window: Optional[int] = None,
    fade_bias: Optional[float] = None,
) -> SceneDetector:
    """Create a scene detector based on the specified type and parameters.

    Args:
        detector_type: Type of detector to create
        threshold: Detection threshold (meaning varies by detector)
        min_scene_len: Minimum scene length in frames
        luma_only: If True, only use brightness for content detection
        adaptive_window: Window size for adaptive detection
        fade_bias: Bias for fade in/out detection (-1.0 to 1.0)

    Note: Parameters set to None will use the detector's built-in default values.

    Returns:
        Configured scene detector instance
    """
    # Set common arguments
    kwargs = {}
    if threshold is not None:
        kwargs["threshold"] = threshold

    if min_scene_len is not None:
        kwargs["min_scene_len"] = min_scene_len

    match detector_type:
        case DetectorType.CONTENT:
            if luma_only is not None:
                kwargs["luma_only"] = luma_only
            return ContentDetector(**kwargs)
        case DetectorType.ADAPTIVE:
            if adaptive_window is not None:
                kwargs["window_width"] = adaptive_window
            if luma_only is not None:
                kwargs["luma_only"] = luma_only
            if "threshold" in kwargs:
                # Special case for adaptive detector which uses different param name
                kwargs["adaptive_threshold"] = kwargs.pop("threshold")
            return AdaptiveDetector(**kwargs)
        case DetectorType.THRESHOLD:
            if fade_bias is not None:
                kwargs["fade_bias"] = fade_bias
            return ThresholdDetector(**kwargs)
        case DetectorType.HISTOGRAM:
            return HistogramDetector(**kwargs)
        case _:
            raise ValueError(f"Unknown detector type: {detector_type}")


def validate_output_dir(output_dir: str) -> Path:
    """Validate and create output directory if it doesn't exist.

    Args:
        output_dir: Path to the output directory

    Returns:
        Path object of the validated output directory
    """
    path = Path(output_dir)

    if path.exists() and not path.is_dir():
        raise argparse.ArgumentTypeError(f"{output_dir} exists but is not a directory")

    return path


def validate_video_file(video_path: str) -> str:
    """Validate that the video file exists.

    Args:
        video_path: Path to the video file

    Returns:
        The validated video path

    Raises:
        argparse.ArgumentTypeError: If file doesn't exist or is a directory
    """
    if not os.path.exists(video_path):
        raise argparse.ArgumentTypeError(f"Video file does not exist: {video_path}")
    
    if os.path.isdir(video_path):
        raise argparse.ArgumentTypeError(f"Path is a directory, not a file: {video_path}")
    
    return video_path


def parse_timecode(video: any, time_str: Optional[str]) -> Optional[FrameTimecode]:
    """Parse a timecode string into a FrameTimecode object.

    Supports formats:
    - Frames: '123'
    - Seconds: '123s' or '123.45s'
    - Timecode: '00:02:03' or '00:02:03.456'

    Args:
        video: Video object to get framerate from
        time_str: String to parse, or None

    Returns:
        FrameTimecode object or None if input is None
    """
    if time_str is None:
        return None

    try:
        if time_str.endswith("s"):
            # Seconds format
            seconds = float(time_str[:-1])
            return FrameTimecode(timecode=seconds, fps=video.frame_rate)
        elif ":" in time_str:
            # Timecode format
            return FrameTimecode(timecode=time_str, fps=video.frame_rate)
        else:
            # Frame number format
            return FrameTimecode(timecode=int(time_str), fps=video.frame_rate)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid timecode format: {time_str}. Use frames (123), "
            f"seconds (123s/123.45s), or timecode (HH:MM:SS[.nnn])",
        ) from e


def detect_and_split_scenes(  # noqa: PLR0913
    video_path: str,
    output_dir: Path,
    detector_type: DetectorType,
    threshold: Optional[float] = None,
    min_scene_len: Optional[int] = None,
    max_scenes: Optional[int] = None,
    filter_shorter_than: Optional[str] = None,
    skip_start: Optional[int] = None,  # noqa: ARG001
    skip_end: Optional[int] = None,  # noqa: ARG001
    save_images_per_scene: int = 0,
    stats_file: Optional[str] = None,
    luma_only: bool = False,
    adaptive_window: Optional[int] = None,
    fade_bias: Optional[float] = None,
    downscale_factor: Optional[int] = None,
    frame_skip: int = 0,
    duration: Optional[str] = None,
) -> List[Tuple[FrameTimecode, FrameTimecode]]:
    """Detect and split scenes in a video using the specified parameters.

    Args:
        video_path: Path to input video.
        output_dir: Directory to save output split scenes.
        detector_type: Type of scene detector to use.
        threshold: Detection threshold.
        min_scene_len: Minimum scene length in frames.
        max_scenes: Maximum number of scenes to detect.
        filter_shorter_than: Filter out scenes shorter than this duration (frames/seconds/timecode)
        skip_start: Number of frames to skip at start.
        skip_end: Number of frames to skip at end.
        save_images_per_scene: Number of images to save per scene (0 to disable).
        stats_file: Path to save detection statistics (optional).
        luma_only: Only use brightness for content detection.
        adaptive_window: Window size for adaptive detection.
        fade_bias: Bias for fade detection (-1.0 to 1.0).
        downscale_factor: Factor to downscale frames by during detection.
        frame_skip: Number of frames to skip (i.e. process every 1 in N+1 frames,
            where N is frame_skip, processing only 1/N+1 percent of the video,
            speeding up the detection time at the expense of accuracy).
            frame_skip must be 0 (the default) when using a StatsManager.
        duration: How much of the video to process from start position.
            Can be specified as frames (123), seconds (123s/123.45s),
            or timecode (HH:MM:SS[.nnn]).

    Returns:
        List of detected scenes as (start, end) FrameTimecode pairs.
    """
    # Create video stream
    video = open_video(video_path, backend="opencv")

    # Parse duration if specified
    duration_tc = parse_timecode(video, duration)

    # Parse filter_shorter_than if specified
    filter_shorter_than_tc = parse_timecode(video, filter_shorter_than)

    # Initialize scene manager with optional stats manager
    stats_manager = StatsManager() if stats_file else None
    scene_manager = SceneManager(stats_manager)

    # Configure scene manager
    if downscale_factor:
        scene_manager.auto_downscale = False
        scene_manager.downscale = downscale_factor

    # Create and add detector
    detector = create_detector(
        detector_type=detector_type,
        threshold=threshold,
        min_scene_len=min_scene_len,
        luma_only=luma_only,
        adaptive_window=adaptive_window,
        fade_bias=fade_bias,
    )
    scene_manager.add_detector(detector)

    # Detect scenes
    print("Detecting scenes...")
    scene_manager.detect_scenes(
        video=video,
        show_progress=True,
        frame_skip=frame_skip,
        duration=duration_tc,
    )

    # Get scene list
    scenes = scene_manager.get_scene_list()

    # Filter out scenes that are too short if filter_shorter_than is specified
    if filter_shorter_than_tc:
        original_count = len(scenes)
        scenes = [
            (start, end)
            for start, end in scenes
            if (end.get_frames() - start.get_frames()) >= filter_shorter_than_tc.get_frames()
        ]
        if len(scenes) < original_count:
            print(
                f"Filtered out {original_count - len(scenes)} scenes shorter "
                f"than {filter_shorter_than_tc.get_seconds():.1f} seconds "
                f"({filter_shorter_than_tc.get_frames()} frames)",
            )

    # Apply max scenes limit if specified
    if max_scenes and len(scenes) > max_scenes:
        print(f"Dropping last {len(scenes) - max_scenes} scenes to meet max_scenes ({max_scenes}) limit")
        scenes = scenes[:max_scenes]

    # Print scene information
    print(f"Found {len(scenes)} scenes:")
    for i, (start, end) in enumerate(scenes, 1):
        print(
            f"Scene {i}: {start.get_timecode()} to {end.get_timecode()} "
            f"({end.get_frames() - start.get_frames()} frames)",
        )

    # Save stats if requested
    if stats_file:
        print(f"Saving detection stats to {stats_file}")
        stats_manager.save_to_csv(stats_file)

    # Split video into scenes
    print("Splitting video into scenes...")
    try:
        split_video_ffmpeg(
            input_video_path=video_path,
            scene_list=scenes,
            output_dir=output_dir,
            show_progress=True,
        )
        print(f"Scenes have been saved to: {output_dir}")
    except Exception as e:
        raise RuntimeError(f"Error splitting video: {e}") from e

    # Save preview images if requested
    if save_images_per_scene > 0:
        print(f"Saving {save_images_per_scene} preview images per scene...")
        image_filenames = save_scene_images(
            scene_list=scenes,
            video=video,
            num_images=save_images_per_scene,
            output_dir=str(output_dir),
            show_progress=True,
        )

        # Generate HTML report with scene information and previews
        html_path = output_dir / "scene_report.html"
        write_scene_list_html(
            output_html_filename=str(html_path),
            scene_list=scenes,
            image_filenames=image_filenames,
        )
        print(f"Scene report saved to: {html_path}")

    return scenes


def main():
    """Main function that sets up argument parsing and runs the scene detection."""
    parser = argparse.ArgumentParser(
        description="Split video into scenes using PySceneDetect.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split video using default content-based detection
  %(prog)s input.mp4 output_dir/

  # Save 3 preview images per scene
  %(prog)s input.mp4 output_dir/ --save-images 3

  # Process specific duration and filter short scenes
  %(prog)s input.mp4 output_dir/ --duration 60s --filter-shorter-than 2s

  # Content detection with minimum scene length and frame skip
  %(prog)s input.mp4 output_dir/ --detector content --min-scene-length 30 --frame-skip 2

  # Use adaptive detection with custom detector and detector parameters
  %(prog)s input.mp4 output_dir/ --detector adaptive --threshold 3.0 --adaptive-window 10
        """
    )

    # Positional arguments
    parser.add_argument(
        "--video_path",
        type=validate_video_file,
        help="Path to the input video file"
    )
    
    parser.add_argument(
        "--output_dir",
        help="Directory where split scenes will be saved"
    )

    # Detection options
    parser.add_argument(
        "--detector",
        choices=[dt.value for dt in DetectorType],
        default=DetectorType.CONTENT.value,
        help="Scene detection algorithm to use (default: %(default)s)"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        help="Detection threshold (meaning varies by detector)"
    )
    
    parser.add_argument(
        "--max_scenes",
        type=int,
        help="Maximum number of scenes to produce"
    )
    
    parser.add_argument(
        "--min_scene_length",
        type=int,
        help="Minimum scene length during detection. Forces the detector to make scenes at least this many frames. "
             "This affects scene detection behavior but does not filter out short scenes."
    )
    
    parser.add_argument(
        "--filter_shorter_than",
        help="Filter out scenes shorter than this duration. Can be specified as frames (123), "
             "seconds (123s/123.45s), or timecode (HH:MM:SS[.nnn]). These scenes will be detected but not saved."
    )

    # Video processing options
    parser.add_argument(
        "--skip_start",
        type=int,
        help="Number of frames to skip at the start of the video"
    )
    
    parser.add_argument(
        "--skip_end",
        type=int,
        help="Number of frames to skip at the end of the video"
    )
    
    parser.add_argument(
        "--duration", "-d",
        help="How much of the video to process. Can be specified as frames (123), "
             "seconds (123s/123.45s), or timecode (HH:MM:SS[.nnn])"
    )
    
    parser.add_argument(
        "--frame_skip",
        type=int,
        default=0,
        help="Number of frames to skip during processing (default: %(default)s)"
    )
    
    parser.add_argument(
        "--downscale",
        type=int,
        help="Factor to downscale frames by during detection"
    )

    # Output options
    parser.add_argument(
        "--save_images",
        type=int,
        default=0,
        help="Number of preview images to save per scene (0 to disable) (default: %(default)s)"
    )
    
    parser.add_argument(
        "--stats_file",
        help="Path to save detection statistics CSV"
    )

    # Detector-specific options
    parser.add_argument(
        "--luma_only",
        action="store_true",
        help="Only use brightness for content detection"
    )
    
    parser.add_argument(
        "--adaptive_window",
        type=int,
        help="Window size for adaptive detection"
    )
    
    parser.add_argument(
        "--fade_bias",
        type=float,
        help="Bias for fade detection (-1.0 to 1.0)"
    )

    args = parser.parse_args()

    # Warning for unsupported features
    if args.skip_start or args.skip_end:
        print("Warning: Skipping start and end frames is not supported yet.")
        return

    # Validate output directory
    try:
        output_path = validate_output_dir(args.output_dir)
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))

    # Convert detector string to enum
    detector_type = DetectorType(args.detector)

    # Detect and split scenes
    try:
        detect_and_split_scenes(
            video_path=args.video_path,
            output_dir=output_path,
            detector_type=detector_type,
            threshold=args.threshold,
            min_scene_len=args.min_scene_length,
            max_scenes=args.max_scenes,
            filter_shorter_than=args.filter_shorter_than,
            skip_start=args.skip_start,
            skip_end=args.skip_end,
            duration=args.duration,
            save_images_per_scene=args.save_images,
            stats_file=args.stats_file,
            luma_only=args.luma_only,
            adaptive_window=args.adaptive_window,
            fade_bias=args.fade_bias,
            downscale_factor=args.downscale,
            frame_skip=args.frame_skip,
        )
    except (RuntimeError, argparse.ArgumentTypeError) as e:
        parser.error(str(e))


if __name__ == "__main__":
    main()
