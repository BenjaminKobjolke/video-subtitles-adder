# Video Subtitles Adder

A Python application that automatically adds subtitles to videos using OpenAI's Whisper speech recognition model.

## Features

- Automatically transcribe audio from videos using OpenAI Whisper
- Support for multiple languages and automatic language detection
- Use existing SRT files if available (same filename as the video)
- Smart resource usage: Whisper model is only loaded when needed
- Customizable subtitle appearance (font, size, color, position)
- Batch processing of multiple videos
- Save separate SRT files (optional)
- Configurable via settings.ini or command-line arguments

## Requirements

- Python 3.8 or higher
- FFmpeg (must be installed and in PATH)
- NVIDIA GPU (optional, for faster processing with CUDA)

## Installation

1. Clone or download this repository
2. Run `install.bat` to set up the virtual environment and install dependencies
3. Run `check_ffmpeg.bat` to verify that FFmpeg is properly installed and working
4. Ensure FFmpeg is installed and in your PATH (if the check fails)

## Usage

### Basic Usage

1. Configure the base directory in `settings.ini` (optional)
2. Place your video files in the input directory
3. Optionally, place SRT files with the same name as the videos in the input directory
   - For example, if you have `video.mp4`, you can place `video.srt` in the same directory
   - The application will use these SRT files instead of transcribing the videos
4. Run `run.bat` to process all videos
5. Find the processed videos in date-based subfolders within the output directory

### Output Organization

For each processed video, the application:

1. Creates a date-based subfolder in the output directory (e.g., `20250315_video_name`)
2. Moves the original video from the input directory to this subfolder
3. Creates the subtitled video with "\_subtitled" suffix in the same subfolder
4. Generates an SRT file in the same subfolder (if enabled in settings)

### Command-line Arguments

```
python -m src.main [options]
```

Options:

- `--config PATH`: Path to the configuration file (default: settings.ini)
- `--input PATH`: Path to a specific input video file
- `--output PATH`: Path to save the output video file
- `--model MODEL`: Whisper model to use (tiny, base, small, medium, large)
- `--language LANG`: Language code for transcription (e.g., en, de, fr)

### Configuration

Edit `settings.ini` to customize the application behavior:

- General settings:
  - `base_dir`: Base directory for input and output (e.g., `e:\test`)
  - `input_dir`: Input directory relative to base_dir (default: `input`)
  - `output_dir`: Output directory relative to base_dir (default: `output`)
- Whisper settings: model size, language, device, maximum segment length
- Subtitle settings: font, size, color, position
- Processing settings: threads, overwrite behavior

## Whisper Models

The application supports different Whisper model sizes:

- `tiny`: Fastest, least accurate (good for quick tests)
- `base`: Fast, reasonable accuracy
- `small`: Balanced speed and accuracy
- `medium`: Good accuracy, slower
- `large`: Best accuracy, slowest

## Subtitle Segmentation

You can control how subtitles are segmented using the following settings in the `[Whisper]` section of settings.ini:

```ini
# Enable word-level timestamps for better segmentation
word_timestamps = true
# Maximum number of lines per subtitle
max_line_count = 1
# Maximum number of characters per line
max_line_width = 20
# Enable sentence-aware segmentation (break at sentence boundaries when possible)
sentence_aware = true
```

These settings help prevent long subtitle segments that might be difficult to read:

- `word_timestamps`: Enables word-level timestamps for more precise segmentation
- `max_line_count`: Limits the number of lines per subtitle segment
- `max_line_width`: Limits the number of characters per line
- `sentence_aware`: When enabled, the application will try to break subtitles at natural sentence boundaries (periods, question marks, exclamation points) rather than strictly adhering to the character limit

### Sentence-Aware Segmentation

The sentence-aware feature intelligently breaks subtitles at natural sentence boundaries. For example, if `max_line_width` is set to 20 and there's a period at character position 18, the subtitle will break at the period rather than continuing to the full 20 characters. This creates more natural-looking subtitles that are easier to read and follow.

If no sentence boundary is found within the maximum line width, the application will fall back to the standard character-based limit.

### Subtitle Quality Improvements

The application includes several features to ensure high-quality subtitles:

1. **Timestamp Validation**: Ensures that subtitle end times are always after start times, fixing any invalid timestamps automatically.

2. **Duplicate Removal**: Removes duplicate subtitle entries to prevent the same text from appearing multiple times.

3. **Overlap Prevention**: Ensures that subtitle entries don't overlap in time, which would cause flickering or simultaneous display of multiple subtitles.

These quality improvements work automatically in the background to produce clean, professional-looking subtitles without requiring any manual intervention.

## GPU Acceleration

The application supports GPU acceleration using CUDA if available:

- Set `device = auto` in settings.ini to automatically use GPU if available (default)
- Set `device = cuda` to force GPU usage (will exit with an error if not available)
- Set `device = cpu` to force CPU usage regardless of GPU availability

Using a GPU can significantly speed up the transcription process, especially for larger models.

## License

This project is open source and available under the MIT License.

## Troubleshooting

### FFmpeg Issues

If the application gets stuck at the "Extracting audio from video..." step:

1. Run `check_ffmpeg.bat` to verify FFmpeg is properly installed and working
2. Check if your video file has an audio track (some videos might be silent)
3. Try processing a different video file to see if the issue is specific to one file
4. Check the logs in the `logs` directory for detailed error messages

### Whisper Transcription Issues

If transcription is slow or fails:

1. Try using a smaller model (e.g., 'tiny' or 'base' instead of 'medium' or 'large')
2. Set `device = cpu` in settings.ini if you're experiencing GPU-related issues
3. For large videos, consider splitting them into smaller segments before processing

### Log Files

The application creates detailed log files in the `logs` directory. These logs contain information about each step of the process and can be helpful for diagnosing issues.

## Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [FFmpeg](https://ffmpeg.org/) for video processing
