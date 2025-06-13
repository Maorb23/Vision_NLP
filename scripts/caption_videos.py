#!/usr/bin/env python3

"""
Auto-caption videos using vision-language models.

This script provides a command-line interface for generating captions for videos using
a vision-language model. It supports processing individual videos or entire directories,
customizing the captioning model, and saving the results to various formats.

The paths to videos in the generated dataset/captions file will be RELATIVE to the
directory where the output file is stored. This makes the dataset more portable and
easier to use in different environments.

Basic usage:
    # Caption a single video
    caption_videos.py video.mp4 --output captions.txt

    # Caption all videos in a directory
    caption_videos.py videos_dir/ --output captions.csv

    # Caption with custom instruction
    caption_videos.py video.mp4 --instruction "Describe what happens in this video in detail."

Advanced usage:
    # Use specific captioner type and device
    caption_videos.py videos_dir/ --captioner-type llava_next_7b --device cuda:0

    # Process videos with specific extensions and save as JSON
    caption_videos.py videos_dir/ --extensions mp4,mov,avi --output captions.json
"""

import argparse
import csv
import json
from enum import Enum
from pathlib import Path

import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from transformers.utils.logging import disable_progress_bar
from tqdm import tqdm
import time

from ltxv_trainer.captioning import (
    DEFAULT_VLM_CAPTION_INSTRUCTION,
    CaptionerType,
    MediaCaptioningModel,
    create_captioner,
)

VIDEO_EXTENSIONS = ["mp4", "avi", "mov", "mkv", "webm"]
IMAGE_EXTENSIONS = ["jpg", "jpeg", "png"]
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS + IMAGE_EXTENSIONS

console = Console()
disable_progress_bar()


class OutputFormat(str, Enum):
    """Available output formats for captions."""

    TXT = "txt"  # Separate files for captions and video paths, one caption / video path per line
    CSV = "csv"  # CSV file with video path and caption columns
    JSON = "json"  # JSON file with video paths as keys and captions as values
    JSONL = "jsonl"  # JSON Lines file with one JSON object per line


def validate_input_path(path_str: str) -> Path:
    """Validate that the input path exists."""
    path = Path(path_str)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Input path does not exist: {path_str}")
    return path


def validate_extensions(extensions_str: str) -> list[str]:
    """Validate and parse the extensions string."""
    try:
        ext_list = [ext.strip() for ext in extensions_str.split(",")]
        return ext_list
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid extensions format: {e}")


def caption_media(
    input_path: Path,
    output_path: Path,
    captioner: MediaCaptioningModel,
    extensions: list[str],
    recursive: bool,
    fps: int,
    clean_caption: bool,
    output_format: OutputFormat,
    override: bool,
) -> None:
    """Caption videos and images using the provided captioning model.
    Args:
        input_path: Path to input video file or directory
        output_path: Path to output caption file
        captioner: Video captioning model
        extensions: List of video file extensions to include
        recursive: Whether to search subdirectories recursively
        fps: Frames per second to sample from videos (ignored for images)
        clean_caption: Whether to clean up captions
        output_format: Format to save the captions in
        override: Whether to override existing captions
    """

    # Get list of media files to process
    media_files = _get_media_files(input_path, extensions, recursive)

    if not media_files:
        console.print("[bold yellow]No media files found to process.[/]")
        return

    console.print(f"Found [bold]{len(media_files)}[/] media files to process.")

    # Get the base directory for relative paths (the directory containing the output file)
    base_dir = output_path.parent.resolve()
    console.print(f"Using [bold blue]{base_dir}[/] as base directory for relative paths")

    # Load existing captions if the output file exists
    existing_captions = _load_existing_captions(output_path, output_format)

    # Convert existing captions keys to absolute paths for comparison
    existing_captions_abs = {}
    for rel_path, caption in existing_captions.items():
        abs_path = str((base_dir / rel_path).resolve())
        existing_captions_abs[abs_path] = caption

    # Filter out media that already have captions if not overriding
    media_to_process = []
    skipped_media = []

    for media_file in media_files:
        media_path_str = str(media_file.resolve())
        if not override and media_path_str in existing_captions_abs:
            skipped_media.append(media_file)
        else:
            media_to_process.append(media_file)

    if skipped_media:
        console.print(f"[bold yellow]Skipping [bold]{len(skipped_media)}[/] media that already have captions.[/]")

    if not media_to_process:
        console.print("[bold yellow]No media to process. All media already have captions.[/]")
        console.print("[bold yellow]Use --override to recaption all media.[/]")
        return

    console.print(f"Processing [bold]{len(media_to_process)}[/] media.")

    # Create progress bar
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
    )

    # Start with existing captions
    captions = existing_captions.copy()

    with progress:
        task = progress.add_task("Generating captions", total=len(media_to_process))

        # In your main processing loop, each "media_file" is now a small scene file
        for media_file in media_to_process:  # These are now scene files, not full videos
            # Update progress description to show current file
            progress.update(task, description=f"Captioning [bold blue]{media_file.name}[/]")

            try:
                # Add more detailed progress info
                console.print(f"[bold cyan]Processing: {media_file.name}[/]")
                console.print(f"[bold yellow]Extracting frames at {fps} FPS and generating caption...[/]")
                
                # Generate caption for the media
                caption = captioner.caption(
                    path=media_file,  # This is now a small scene file
                    fps=fps,
                    clean_caption=clean_caption,
                )
                
                console.print(f"[bold green]✓ Generated caption for {media_file.name}[/]")
                console.print(f"[bold dim]Caption preview: {caption[:100]}...[/]")

                # Convert absolute path to relative path (relative to the output file's directory)
                rel_path = str(media_file.resolve().relative_to(base_dir))
                # Store the caption with the relative path as key
                captions[rel_path] = caption

                # 💾 Save after each scene (much more frequent saves)
                _save_captions(captions, output_path, output_format)
                console.print(f"[bold dim]💾 Saved progress ({len(captions)} scenes)[/]")
                
            except Exception as e:
                console.print(f"[bold red]Error captioning scene {media_file}: {e}[/]")

            # Advance progress bar
            progress.advance(task)

    # Save captions to file
    _save_captions(captions, output_path, output_format)

    # Print summary
    processed_media = len(captions) - len(existing_captions)
    total_to_process = len(media_files) - len(skipped_media)
    console.print(
        f"[bold green]✓[/] Captioned [bold]{processed_media}/{total_to_process}[/] media successfully.",
    )


def _get_media_files(
    input_path: Path,
    extensions: list[str] = MEDIA_EXTENSIONS,
    recursive: bool = False,
) -> list[Path]:
    """Get all media files from the input path."""
    input_path = Path(input_path)
    # Normalize extensions to lowercase without dots
    extensions = [ext.lower().lstrip(".") for ext in extensions]

    if input_path.is_file():
        # If input is a file, check if it has a valid extension
        if input_path.suffix.lstrip(".").lower() in extensions:
            return [input_path]
        else:
            console.print(f"[bold yellow]Warning: {input_path} is not a recognized media file. Skipping.[/]")
            return []
    elif input_path.is_dir():
        # If input is a directory, find all media files
        media_files = []

        # Define the glob pattern based on whether we're searching recursively
        glob_pattern = "**/*" if recursive else "*"

        # Find all files with the specified extensions
        for ext in extensions:
            media_files.extend(input_path.glob(f"{glob_pattern}.{ext}"))

        return sorted(media_files)
    else:
        raise argparse.ArgumentTypeError(f"Error: {input_path} does not exist.")


def _save_captions(
    captions: dict[str, str],
    output_path: Path,
    format_type: OutputFormat,
) -> None:
    """Save captions to a file in the specified format.

    Args:
        captions: Dictionary mapping media paths to captions
        output_path: Path to save the output file
        format_type: Format to save the captions in
    """
    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    console.print("[bold blue]Saving captions...[/]")

    match format_type:
        case OutputFormat.TXT:
            # Create two separate files for captions and media paths
            captions_file = output_path.with_stem(f"{output_path.stem}_captions")
            paths_file = output_path.with_stem(f"{output_path.stem}_paths")

            with captions_file.open("w", encoding="utf-8") as f:
                for caption in captions.values():
                    f.write(f"{caption}\n")

            with paths_file.open("w", encoding="utf-8") as f:
                for media_path in captions:
                    f.write(f"{media_path}\n")

            console.print(f"[bold green]✓[/] Captions saved to [cyan]{captions_file}[/]")
            console.print(f"[bold green]✓[/] Media paths saved to [cyan]{paths_file}[/]")
            console.print("[bold yellow]Note:[/] Use these files with ImageOrVideoDataset by setting:")
            console.print(f"  caption_column='{captions_file.name}'")
            console.print(f"  video_column='{paths_file.name}'")

        case OutputFormat.CSV:
            with output_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["caption", "media_path"])
                for media_path, caption in captions.items():
                    writer.writerow([caption, media_path])

            console.print(f"[bold green]✓[/] Captions saved to [cyan]{output_path}[/]")
            console.print("[bold yellow]Note:[/] Use these files with ImageOrVideoDataset by setting:")
            console.print("  caption_column='[cyan]caption[/]'")
            console.print("  video_column='[cyan]media_path[/]'")

        case OutputFormat.JSON:
            # Format as list of dictionaries with caption and media_path keys
            json_data = [{"caption": caption, "media_path": media_path} for media_path, caption in captions.items()]

            with output_path.open("w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            console.print(f"[bold green]✓[/] Captions saved to [cyan]{output_path}[/]")
            console.print("[bold yellow]Note:[/] Use these files with ImageOrVideoDataset by setting:")
            console.print("  caption_column='[cyan]caption[/]'")
            console.print("  video_column='[cyan]media_path[/]'")

        case OutputFormat.JSONL:
            with output_path.open("w", encoding="utf-8") as f:
                for media_path, caption in captions.items():
                    f.write(json.dumps({"caption": caption, "media_path": media_path}, ensure_ascii=False) + "\n")

            console.print(f"[bold green]✓[/] Captions saved to [cyan]{output_path}[/]")
            console.print("[bold yellow]Note:[/] Use these files with ImageOrVideoDataset by setting:")
            console.print("  caption_column='[cyan]caption[/]'")
            console.print("  video_column='[cyan]media_path[/]'")

        case _:
            raise ValueError(f"Unsupported output format: {format_type}")


def _load_existing_captions(  # noqa: PLR0912
    output_path: Path,
    format_type: OutputFormat,
) -> dict[str, str]:
    """Load existing captions from a file.

    Args:
        output_path: Path to the captions file
        format_type: Format of the captions file

    Returns:
        Dictionary mapping media paths to captions, or empty dict if file doesn't exist
    """
    if not output_path.exists():
        return {}

    console.print(f"[bold blue]Loading existing captions from [cyan]{output_path}[/]...[/]")

    existing_captions = {}

    try:
        match format_type:
            case OutputFormat.TXT:
                # For TXT format, we have two separate files
                captions_file = output_path.with_stem(f"{output_path.stem}_captions")
                paths_file = output_path.with_stem(f"{output_path.stem}_paths")

                if captions_file.exists() and paths_file.exists():
                    captions = captions_file.read_text(encoding="utf-8").splitlines()
                    paths = paths_file.read_text(encoding="utf-8").splitlines()

                    if len(captions) == len(paths):
                        existing_captions = dict(zip(paths, captions, strict=False))

            case OutputFormat.CSV:
                with output_path.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.reader(f)
                    # Skip header
                    next(reader, None)
                    for row in reader:
                        if len(row) >= 2:
                            caption, media_path = row[0], row[1]
                            existing_captions[media_path] = caption

            case OutputFormat.JSON:
                with output_path.open("r", encoding="utf-8") as f:
                    json_data = json.load(f)
                    for item in json_data:
                        if "caption" in item and "media_path" in item:
                            existing_captions[item["media_path"]] = item["caption"]

            case OutputFormat.JSONL:
                with output_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        item = json.loads(line)
                        if "caption" in item and "media_path" in item:
                            existing_captions[item["media_path"]] = item["caption"]

            case _:
                raise ValueError(f"Unsupported output format: {format_type}")

        console.print(f"[bold green]✓[/] Loaded [bold]{len(existing_captions)}[/] existing captions")
        return existing_captions

    except Exception as e:
        console.print(f"[bold yellow]Warning: Could not load existing captions: {e}[/]")
        return {}


def create_captioner_with_progress(
    captioner_type: CaptionerType,
    device: str,
    use_8bit: bool,
    vlm_instruction: str,
) -> MediaCaptioningModel:
    """Create captioner with progress tracking for model loading."""
    
    console.print("[bold blue]Initializing captioning model...[/]")
    console.print(f"[bold yellow]Device: {device}[/]")
    console.print(f"[bold yellow]Model: {captioner_type}[/]")
    console.print(f"[bold yellow]8-bit: {use_8bit}[/]")
    
    # Enable transformers progress bars (this shows the real progress you saw)
    from transformers.utils.logging import enable_progress_bar, set_verbosity_info
    enable_progress_bar()
    set_verbosity_info()
    
    start_time = time.time()
    
    console.print("[bold cyan]Starting model download and loading...[/]")
    console.print("[bold yellow]This may take several minutes on first run![/]")
    
    try:
        # This is where the real loading happens - no fake tqdm needed
        captioner = create_captioner(
            captioner_type=captioner_type,
            device=device,
            use_8bit=use_8bit,
            vlm_instruction=vlm_instruction,
        )
        
    except Exception as e:
        console.print(f"[bold red]✗ Error creating captioner: {e}[/]")
        raise e
    
    # Disable progress bars for cleaner captioning output
    disable_progress_bar()
    
    elapsed_time = time.time() - start_time
    console.print(f"[bold green]✓ Model loaded successfully in {elapsed_time:.1f} seconds[/]")
    
    return captioner


def main():
    """Main function that sets up argument parsing and runs the caption generation."""
    parser = argparse.ArgumentParser(
        description="Auto-caption videos using vision-language models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Caption a single video with default settings
  %(prog)s scenes_output_dir/ --output scenes_output_dir/captions.json

  # Caption using specific captioner type
  %(prog)s scenes_output_dir/ --output captions.json --captioner-type llava_next_7b

  # Caption with custom instruction and device
  %(prog)s scenes_output_dir/ --output captions.json --captioner-type qwen_25_vl --device cuda:0 --instruction "Describe this video in detail"

  # Process with specific extensions and recursive search
  %(prog)s video_dir/ --output captions.csv --extensions mp4,mov,avi --recursive

  # Override existing captions with 8-bit precision
  %(prog)s video_dir/ --output captions.json --override --use-8bit

Valid captioner types: qwen_25_vl, llava_next_7b
Valid output formats: json, csv, txt, jsonl (determined by file extension)
        """
    )

    # Positional argument
    parser.add_argument(
        "input_path",
        type=validate_input_path,
        help="Path to input video/image file or directory containing media files"
    )

    # Required arguments
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Path to output file for captions. Format determined by file extension. If not specified, defaults to dataset.json in input directory."
    )

    # Model options
    parser.add_argument(
        "--captioner-type", "-c",
        choices=[ct.value for ct in CaptionerType],
        default=CaptionerType.QWEN_25_VL.value,
        help="Type of captioner to use (default: %(default)s)"
    )

    parser.add_argument(
        "--device", "-d",
        help="Device to use for inference (e.g., 'cuda', 'cuda:0', 'cpu'). Auto-detected if not specified."
    )

    parser.add_argument(
        "--use-8bit",
        action="store_true",
        help="Whether to use 8-bit precision for the captioning model"
    )

    parser.add_argument(
        "--instruction", "-i",
        default=DEFAULT_VLM_CAPTION_INSTRUCTION,
        help="Instruction to give to the captioning model"
    )

    # Media processing options
    parser.add_argument(
        "--extensions", "-e",
        type=validate_extensions,
        default=MEDIA_EXTENSIONS,
        help=f"Comma-separated list of media file extensions to process (default: {','.join(MEDIA_EXTENSIONS)})"
    )

    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Search for media files in subdirectories recursively"
    )

    parser.add_argument(
        "--fps", "-f",
        type=int,
        default=3,
        help="Frames per second to sample from videos (ignored for images) (default: %(default)s)"
    )

    parser.add_argument(
        "--clean-caption",
        action="store_true",
        default=True,
        help="Whether to clean up captions by removing common VLM patterns (default: %(default)s)"
    )

    parser.add_argument(
        "--no-clean-caption",
        dest="clean_caption",
        action="store_false",
        help="Disable caption cleaning"
    )

    parser.add_argument(
        "--override",
        action="store_true",
        help="Whether to override existing captions for media"
    )

    args = parser.parse_args()

    # Determine device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Determine output path and format
    if args.output is None:
        if args.input_path.is_file():
            # Default to a JSON file with the same name as the input media
            output_path = args.input_path.with_suffix(".dataset.json")
        else:
            # Default to a JSON file in the input directory
            output_path = args.input_path / "dataset.json"
    else:
        output_path = args.output

    # Determine format from file extension
    output_format = OutputFormat(Path(output_path).suffix.lstrip(".").lower())

    # Ensure output path is absolute
    output_path = Path(output_path).resolve()
    console.print(f"Output will be saved to [bold blue]{output_path}[/]")

    # Initialize captioning model with progress tracking
    captioner = create_captioner_with_progress(
        captioner_type=CaptionerType(args.captioner_type),
        device=device,
        use_8bit=args.use_8bit,
        vlm_instruction=args.instruction,
    )

    # Caption media files
    try:
        caption_media(
            input_path=args.input_path,
            output_path=output_path,
            captioner=captioner,
            extensions=args.extensions,
            recursive=args.recursive,
            fps=args.fps,
            clean_caption=args.clean_caption,
            output_format=output_format,
            override=args.override,
        )
    except Exception as e:
        parser.error(f"Error during captioning: {e}")


if __name__ == "__main__":
    main()

