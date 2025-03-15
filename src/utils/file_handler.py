"""
File handler module for the Video Subtitles Adder application.
Provides functionality for file operations like scanning directories,
checking file types, and managing output paths.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional

class FileHandler:
    """
    Handles file operations for the Video Subtitles Adder application.
    """
    
    # Supported video file extensions
    SUPPORTED_VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    def __init__(self, input_dir: str = "input", output_dir: str = "output"):
        """
        Initialize the file handler.
        
        Args:
            input_dir: The input directory path.
            output_dir: The output directory path.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_video_files(self) -> List[str]:
        """
        Get a list of all video files in the input directory.
        
        Returns:
            A list of video file paths.
        """
        video_files = []
        
        # Check if input directory exists
        if not os.path.exists(self.input_dir):
            raise FileNotFoundError(f"Input directory '{self.input_dir}' not found.")
        
        # Scan input directory for video files
        for file in os.listdir(self.input_dir):
            file_path = os.path.join(self.input_dir, file)
            
            # Check if it's a file and has a supported extension
            if os.path.isfile(file_path) and self._is_video_file(file_path):
                video_files.append(file_path)
        
        return video_files
    
    def create_output_subfolder(self, input_path: str) -> str:
        """
        Create a date-based subfolder in the output directory for a given input file.
        
        Args:
            input_path: The input file path.
        
        Returns:
            The path to the created subfolder.
        """
        # Get the filename without extension
        filename = os.path.basename(input_path)
        name, _ = os.path.splitext(filename)
        
        # Create subfolder name with date prefix
        date_str = datetime.now().strftime("%Y%m%d")
        subfolder_name = f"{date_str}_{name}"
        
        # Create full subfolder path
        subfolder_path = os.path.join(self.output_dir, subfolder_name)
        
        # Create the subfolder
        os.makedirs(subfolder_path, exist_ok=True)
        
        return subfolder_path
    
    def move_original_to_output(self, input_path: str, output_subfolder: str) -> str:
        """
        Move the original input file to the output subfolder.
        
        Args:
            input_path: The input file path.
            output_subfolder: The output subfolder path.
        
        Returns:
            The path to the moved file.
        """
        # Get the filename
        filename = os.path.basename(input_path)
        
        # Create the destination path
        dest_path = os.path.join(output_subfolder, filename)
        
        # Move the file
        shutil.move(input_path, dest_path)
        
        return dest_path
    
    def get_output_path(self, input_path: str, suffix: str = "_subtitled") -> str:
        """
        Generate the output file path for a given input file.
        
        Args:
            input_path: The input file path.
            suffix: The suffix to add to the output filename.
        
        Returns:
            The output file path.
        """
        # Get the filename and extension
        filename = os.path.basename(input_path)
        name, ext = os.path.splitext(filename)
        
        # Create the output filename
        output_filename = f"{name}{suffix}{ext}"
        
        # Create output subfolder
        output_subfolder = self.create_output_subfolder(input_path)
        
        # Return the full output path
        return os.path.join(output_subfolder, output_filename)
    
    def get_srt_path(self, input_path: str) -> str:
        """
        Generate the SRT file path for a given input file.
        
        Args:
            input_path: The input file path.
        
        Returns:
            The SRT file path.
        """
        # Get the filename and extension
        filename = os.path.basename(input_path)
        name, _ = os.path.splitext(filename)
        
        # Create the SRT filename
        srt_filename = f"{name}.srt"
        
        # Create output subfolder
        output_subfolder = self.create_output_subfolder(input_path)
        
        # Return the full SRT path
        return os.path.join(output_subfolder, srt_filename)
    
    def check_existing_srt(self, video_path: str) -> Optional[str]:
        """
        Check if there's an existing SRT file with the same name as the video.
        
        Args:
            video_path: The path to the video file.
        
        Returns:
            The path to the existing SRT file, or None if it doesn't exist.
        """
        # Get the filename without extension
        filename = os.path.basename(video_path)
        name, _ = os.path.splitext(filename)
        
        # Create the SRT filename
        srt_filename = f"{name}.srt"
        
        # Get the directory of the video file
        video_dir = os.path.dirname(video_path)
        
        # Create the full SRT path
        srt_path = os.path.join(video_dir, srt_filename)
        
        # Check if the SRT file exists
        if os.path.exists(srt_path):
            return srt_path
        
        return None
    
    def _is_video_file(self, file_path: str) -> bool:
        """
        Check if a file is a supported video file.
        
        Args:
            file_path: The file path to check.
        
        Returns:
            True if the file is a supported video file, False otherwise.
        """
        _, ext = os.path.splitext(file_path.lower())
        return ext in self.SUPPORTED_VIDEO_EXTENSIONS
