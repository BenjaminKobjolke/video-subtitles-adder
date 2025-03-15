"""
Subtitler module for the Video Subtitles Adder application.
Handles adding subtitles to videos.
"""

import os
import subprocess
from typing import Dict, Any, List, Optional

from src.core.transcriber import Transcription
from src.utils.logger import Logger

class Subtitler:
    """
    Handles adding subtitles to videos.
    """
    
    def __init__(self, subtitle_settings: Dict[str, Any], logger: Logger):
        """
        Initialize the subtitler.
        
        Args:
            subtitle_settings: The subtitle settings.
            logger: The logger instance.
        """
        self.settings = subtitle_settings
        self.logger = logger
    
    def add_subtitles(self, video_path: str, srt_path: str, output_path: str):
        """
        Add subtitles to a video.
        
        Args:
            video_path: The path to the input video.
            srt_path: The path to the SRT file.
            output_path: The path to save the output video.
        """
        self.logger.info(f"Adding subtitles to '{video_path}'...")
        
        # Check if input files exist
        if not os.path.exists(video_path):
            self.logger.error(f"Input video not found: '{video_path}'")
            raise FileNotFoundError(f"Input video '{video_path}' not found.")
        else:
            self.logger.info(f"Input video exists: '{video_path}' ({os.path.getsize(video_path) / (1024*1024):.2f} MB)")
        
        if not os.path.exists(srt_path):
            self.logger.error(f"SRT file not found: '{srt_path}'")
            raise FileNotFoundError(f"SRT file '{srt_path}' not found.")
        else:
            self.logger.info(f"SRT file exists: '{srt_path}' ({os.path.getsize(srt_path) / 1024:.2f} KB)")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Output directory ensured: '{output_dir}'")
        
        # Build ffmpeg command
        command = self._build_ffmpeg_command(video_path, srt_path, output_path)
        
        try:
            # Run the command without capturing output to see FFmpeg progress
            self.logger.info("Running FFmpeg to add subtitles...")
            process = subprocess.run(
                command,
                check=True,
                text=True,
                stdout=subprocess.PIPE,  # Capture stdout but still show it
                stderr=subprocess.PIPE   # Capture stderr but still show it
            )
            
            # Check if the output file was created
            if os.path.exists(output_path):
                output_size = os.path.getsize(output_path) / (1024*1024)
                self.logger.info(f"Subtitles added successfully. Output saved to '{output_path}' ({output_size:.2f} MB)")
            else:
                self.logger.error(f"FFmpeg completed but output file was not created: '{output_path}'")
                raise RuntimeError(f"Failed to add subtitles: Output file was not created")
        
        except subprocess.CalledProcessError as e:
            self.logger.error(f"FFmpeg process failed with return code {e.returncode}")
            if e.stderr:
                self.logger.error(f"FFmpeg error output: {e.stderr}")
            
            # Try to provide more helpful error information
            if "No such file or directory" in e.stderr:
                self.logger.error("This may be due to path issues with the SRT file or output directory")
            
            raise RuntimeError(f"Failed to add subtitles: FFmpeg process failed")
    
    def _build_ffmpeg_command(self, video_path: str, srt_path: str, output_path: str) -> List[str]:
        """
        Build the FFmpeg command to add subtitles to a video.
        
        Args:
            video_path: The path to the input video.
            srt_path: The path to the SRT file.
            output_path: The path to save the output video.
        
        Returns:
            The FFmpeg command as a list of strings.
        """
        # Get subtitle settings
        font = self.settings['font']
        font_size = self.settings['font_size']
        font_color = self.settings['font_color']
        outline_color = self.settings['outline_color']
        outline_width = self.settings['outline_width']
        position = self.settings['position']
        burn_subtitles = self.settings.get('burn_subtitles', False)
        
        # Determine vertical position
        if position == 'top':
            vertical_position = '10'
        elif position == 'middle':
            vertical_position = '(h-text_h)/2'
        else:  # bottom (default)
            vertical_position = 'h-text_h-10'
        
        # Log the paths for debugging
        self.logger.info(f"Video path: {video_path}")
        self.logger.info(f"SRT path: {srt_path}")
        self.logger.info(f"Output path: {output_path}")
        self.logger.info(f"Burn subtitles: {burn_subtitles}")
        
        if burn_subtitles:
            # Burn subtitles directly into the video using the simple subtitles filter
            # Based on FFmpeg documentation: ffmpeg -i video.avi -vf subtitles=subtitle.srt out.avi
            
            # Use the simple subtitles filter syntax with properly escaped path
            # Convert backslashes to forward slashes and escape special characters
            escaped_srt_path = srt_path.replace('\\', '/').replace(':', '\\:')
            
            # Use the -copyts option to preserve timestamps
            command = [
                "ffmpeg",
                "-i", video_path,
                "-vf", f"subtitles='{escaped_srt_path}'",  # Quoted and escaped path
                "-c:v", "libx264",
                "-crf", "18",
                "-c:a", "copy",
                "-y",  # Overwrite output file if it exists
                output_path
            ]
            
            self.logger.info("Using burned-in subtitles (permanently embedded in video)")
        else:
            # Add subtitles as a separate track (can be turned on/off)
            command = [
                "ffmpeg",
                "-i", video_path,
                "-f", "srt",
                "-i", srt_path,
                "-map", "0:v",
                "-map", "0:a",
                "-map", "1",
                "-c:v", "libx264",
                "-crf", "18",
                "-c:a", "copy",
                "-c:s", "mov_text",
                "-metadata:s:s:0", f"language=eng",
                "-y",  # Overwrite output file if it exists
                output_path
            ]
            
            self.logger.info("Using subtitle track (can be turned on/off in video players)")
        
        # Log the full command for debugging
        self.logger.info(f"FFmpeg command: {' '.join(command)}")
        
        return command
    
    def _color_to_hex(self, color: str) -> str:
        """
        Convert a color name or RGB value to FFmpeg hex format.
        
        Args:
            color: The color name or RGB value.
        
        Returns:
            The color in FFmpeg hex format.
        """
        # Common color names
        color_map = {
            'white': 'FFFFFF',
            'black': '000000',
            'red': 'FF0000',
            'green': '00FF00',
            'blue': '0000FF',
            'yellow': 'FFFF00',
            'cyan': '00FFFF',
            'magenta': 'FF00FF',
        }
        
        # Check if color is in the map
        if color.lower() in color_map:
            return color_map[color.lower()]
        
        # Check if color is a hex value
        if color.startswith('#'):
            return color[1:]
        
        # Default to white
        return 'FFFFFF'
