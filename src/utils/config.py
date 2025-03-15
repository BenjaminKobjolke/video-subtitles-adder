"""
Configuration module for the Video Subtitles Adder application.
Handles reading and parsing the settings.ini file.
"""

import configparser
import os
from typing import Dict, Any, List, Optional

class Config:
    """
    Configuration class for the Video Subtitles Adder application.
    Reads and parses the settings.ini file.
    """
    
    def __init__(self, config_file: str = "settings.ini"):
        """
        Initialize the configuration.
        
        Args:
            config_file: The path to the configuration file.
        """
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        
        # Check if config file exists
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file '{config_file}' not found.")
        
        # Read the config file
        self.config.read(config_file)
    
    def get_general_settings(self) -> Dict[str, Any]:
        """
        Get the general settings.
        
        Returns:
            A dictionary of general settings.
        """
        # Get base directory
        base_dir = self.config.get('General', 'base_dir', fallback='')
        
        # Get input and output directories
        input_dir = self.config.get('General', 'input_dir', fallback='input')
        output_dir = self.config.get('General', 'output_dir', fallback='output')
        
        # If base_dir is specified, join it with input_dir and output_dir
        if base_dir:
            input_dir = os.path.join(base_dir, input_dir)
            output_dir = os.path.join(base_dir, output_dir)
        
        return {
            'base_dir': base_dir,
            'input_dir': input_dir,
            'output_dir': output_dir,
            'process_all': self.config.getboolean('General', 'process_all', fallback=True)
        }
    
    def get_whisper_settings(self) -> Dict[str, Any]:
        """
        Get the Whisper settings.
        
        Returns:
            A dictionary of Whisper settings.
        """
        # Parse language list if provided
        language_str = self.config.get('Whisper', 'language', fallback='')
        languages = [lang.strip() for lang in language_str.split(',')] if language_str else []
        
        # Get device setting
        device = self.config.get('Whisper', 'device', fallback='auto')
        
        # Handle 'auto' device setting
        if device == 'auto':
            try:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                device = 'cpu'
        
        return {
            'model': self.config.get('Whisper', 'model', fallback='base'),
            'device': device,
            'languages': languages,
            'detect_language': self.config.getboolean('Whisper', 'detect_language', fallback=True),
            'word_timestamps': self.config.getboolean('Whisper', 'word_timestamps', fallback=True),
            'max_line_count': self.config.getint('Whisper', 'max_line_count', fallback=1),
            'max_line_width': self.config.getint('Whisper', 'max_line_width', fallback=20),
            'sentence_aware': self.config.getboolean('Whisper', 'sentence_aware', fallback=True)
        }
    
    def get_subtitle_settings(self) -> Dict[str, Any]:
        """
        Get the subtitle settings.
        
        Returns:
            A dictionary of subtitle settings.
        """
        return {
            'font': self.config.get('Subtitles', 'font', fallback='Arial'),
            'font_size': self.config.getint('Subtitles', 'font_size', fallback=24),
            'font_color': self.config.get('Subtitles', 'font_color', fallback='white'),
            'outline_color': self.config.get('Subtitles', 'outline_color', fallback='black'),
            'outline_width': self.config.getint('Subtitles', 'outline_width', fallback=1),
            'position': self.config.get('Subtitles', 'position', fallback='bottom'),
            'max_chars_per_line': self.config.getint('Subtitles', 'max_chars_per_line', fallback=40),
            'save_srt': self.config.getboolean('Subtitles', 'save_srt', fallback=True),
            'burn_subtitles': self.config.getboolean('Subtitles', 'burn_subtitles', fallback=False)
        }
    
    def get_processing_settings(self) -> Dict[str, Any]:
        """
        Get the processing settings.
        
        Returns:
            A dictionary of processing settings.
        """
        return {
            'threads': self.config.getint('Processing', 'threads', fallback=0),
            'overwrite': self.config.getboolean('Processing', 'overwrite', fallback=False),
            'show_progress': self.config.getboolean('Processing', 'show_progress', fallback=True)
        }
    
    def get_all_settings(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all settings.
        
        Returns:
            A dictionary of all settings.
        """
        return {
            'general': self.get_general_settings(),
            'whisper': self.get_whisper_settings(),
            'subtitles': self.get_subtitle_settings(),
            'processing': self.get_processing_settings()
        }
