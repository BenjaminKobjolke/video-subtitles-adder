"""
Main module for the Video Subtitles Adder application.
Handles command-line arguments and orchestrates the application flow.
"""

import os
import sys
import argparse
import time
import shutil
from typing import List, Optional

from src.utils.logger import Logger
from src.utils.config import Config
from src.utils.file_handler import FileHandler
from src.core.transcriber import Transcriber
from src.core.subtitler import Subtitler

def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Add subtitles to videos using OpenAI Whisper.")
    
    parser.add_argument(
        "--config",
        type=str,
        default="settings.ini",
        help="Path to the configuration file (default: settings.ini)"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        help="Path to the input video file (overrides config file)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Path to the output video file (overrides config file)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model to use (overrides config file)"
    )
    
    parser.add_argument(
        "--language",
        type=str,
        help="Language code for transcription (overrides config file)"
    )
    
    return parser.parse_args()

def process_video(
    video_path: str,
    transcriber: Transcriber,
    subtitler: Subtitler,
    file_handler: FileHandler,
    save_srt: bool,
    logger: Logger
):
    """
    Process a single video.
    
    Args:
        video_path: The path to the video file.
        transcriber: The transcriber instance.
        subtitler: The subtitler instance.
        file_handler: The file handler instance.
        save_srt: Whether to save the SRT file.
        logger: The logger instance.
    """
    try:
        logger.info(f"Processing video: {video_path}")
        
        # Check if there's an existing SRT file with the same name
        existing_srt = file_handler.check_existing_srt(video_path)
        
        # Generate output paths
        output_path = file_handler.get_output_path(video_path)
        srt_path = file_handler.get_srt_path(video_path)
        
        # Get the output subfolder path (extract from output_path)
        output_subfolder = os.path.dirname(output_path)
        
        # Move original file to output subfolder
        moved_path = file_handler.move_original_to_output(video_path, output_subfolder)
        logger.info(f"Moved original video to: {moved_path}")
        
        # If there's an existing SRT file, use it instead of transcribing
        if existing_srt:
            logger.info(f"Found existing SRT file: {existing_srt}")
            
            # Move the existing SRT file to the output subfolder
            srt_filename = os.path.basename(existing_srt)
            dest_srt_path = os.path.join(output_subfolder, srt_filename)
            shutil.move(existing_srt, dest_srt_path)
            logger.info(f"Moved existing SRT file to: {dest_srt_path}")
            
            # Update the srt_path to point to the moved file
            srt_path = dest_srt_path
        else:
            # No existing SRT file, so transcribe the video
            logger.info("No existing SRT file found. Transcribing video...")
            
            # Check if transcriber is available
            if transcriber is None:
                logger.error("Transcriber is not initialized but an SRT file is needed.")
                raise RuntimeError("Cannot transcribe video: Whisper model not initialized")
            
            # Transcribe the video using the moved file path
            transcription = transcriber.transcribe(moved_path)
            
            # Save SRT file
            transcriber.save_srt(transcription, srt_path)
        
        # Add subtitles to the video
        subtitler.add_subtitles(moved_path, srt_path, output_path)
        
        # Clean up SRT file if not saving
        if not save_srt and os.path.exists(srt_path):
            os.remove(srt_path)
            logger.info(f"Removed temporary SRT file: {srt_path}")
        
        logger.info(f"Video processing completed: {output_path}")
    
    except Exception as e:
        logger.error(f"Failed to process video '{video_path}': {str(e)}")
        raise

def main():
    """
    Main entry point for the application.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Initialize logger
    logger = Logger()
    
    try:
        # Load configuration
        config = Config(args.config)
        settings = config.get_all_settings()
        
        # Override settings with command-line arguments
        if args.input:
            settings['general']['input_dir'] = os.path.dirname(args.input)
        
        if args.output:
            settings['general']['output_dir'] = os.path.dirname(args.output)
        
        if args.model:
            settings['whisper']['model'] = args.model
        
        if args.language:
            settings['whisper']['languages'] = [args.language]
            settings['whisper']['detect_language'] = False
        
        # Initialize file handler
        file_handler = FileHandler(
            input_dir=settings['general']['input_dir'],
            output_dir=settings['general']['output_dir']
        )
        
        # Get video files to process
        if args.input and os.path.isfile(args.input):
            video_files = [args.input]
        else:
            video_files = file_handler.get_video_files()
        
        if not video_files:
            logger.warning(f"No video files found in '{settings['general']['input_dir']}'.")
            return
        
        logger.info(f"Found {len(video_files)} video file(s) to process.")
        
        # Check if all videos have matching SRT files
        all_have_srt = True
        for video_path in video_files:
            if not file_handler.check_existing_srt(video_path):
                all_have_srt = False
                break
        
        # Initialize transcriber only if needed
        transcriber = None
        if not all_have_srt:
            logger.info("Some videos don't have matching SRT files. Initializing Whisper model...")
            transcriber = Transcriber(
                whisper_settings=settings['whisper'],
                logger=logger
            )
        else:
            logger.info("All videos have matching SRT files. Skipping Whisper model initialization.")
        
        # Initialize subtitler
        logger.info("Initializing subtitler...")
        subtitler = Subtitler(
            subtitle_settings=settings['subtitles'],
            logger=logger
        )
        
        # Process each video
        start_time = time.time()
        
        for i, video_path in enumerate(video_files, 1):
            logger.info(f"Processing video {i}/{len(video_files)}: {video_path}")
            
            process_video(
                video_path=video_path,
                transcriber=transcriber,
                subtitler=subtitler,
                file_handler=file_handler,
                save_srt=settings['subtitles']['save_srt'],
                logger=logger
            )
        
        # Log completion
        elapsed_time = time.time() - start_time
        logger.info(f"All videos processed successfully in {elapsed_time:.2f} seconds.")
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
