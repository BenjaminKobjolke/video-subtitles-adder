"""
Transcriber module for the Video Subtitles Adder application.
Uses OpenAI Whisper to transcribe audio from videos.
"""

import os
import tempfile
import subprocess
from typing import Dict, Any, List, Optional, Tuple

import whisper
from whisper.utils import get_writer

from src.utils.logger import Logger

class Transcription:
    """
    Class to store transcription data.
    """
    
    def __init__(self, segments: List[Dict[str, Any]], language: str):
        """
        Initialize the transcription.
        
        Args:
            segments: The transcription segments.
            language: The detected language.
        """
        self.segments = segments
        self.language = language
    
    def get_segments(self) -> List[Dict[str, Any]]:
        """
        Get the transcription segments.
        
        Returns:
            The transcription segments.
        """
        return self.segments
    
    def get_language(self) -> str:
        """
        Get the detected language.
        
        Returns:
            The detected language.
        """
        return self.language


class Transcriber:
    """
    Handles transcription of audio from videos using OpenAI Whisper.
    """
    
    def __init__(self, whisper_settings: Dict[str, Any], logger: Logger):
        """
        Initialize the transcriber.
        
        Args:
            whisper_settings: The Whisper settings.
            logger: The logger instance.
        """
        self.settings = whisper_settings
        self.logger = logger
        self.model = None
        
        # Load the Whisper model
        self._load_model()
    
    def _load_model(self):
        """
        Load the Whisper model.
        """
        model_name = self.settings['model']
        requested_device = self.settings['device']
        
        # Check if CUDA is available when 'cuda' is requested
        if requested_device == 'cuda':
            import torch
            if not torch.cuda.is_available():
                self.logger.error("CUDA requested but not available. Exiting.")
                raise RuntimeError("CUDA (GPU) was explicitly requested in settings but is not available on this system. Please set device = 'auto' or device = 'cpu' in settings.ini.")
            device = 'cuda'
        else:
            device = requested_device
        
        self.logger.info(f"Loading Whisper model '{model_name}' on device '{device}'...")
        
        try:
            self.model = whisper.load_model(model_name, device=device)
            self.logger.info(f"Whisper model '{model_name}' loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {str(e)}")
            raise
    
    def transcribe(self, video_path: str) -> Transcription:
        """
        Transcribe audio from a video file.
        
        Args:
            video_path: The path to the video file.
        
        Returns:
            A Transcription object containing the transcription data.
        """
        import time
        overall_start_time = time.time()
        
        self.logger.info(f"Transcribing audio from '{video_path}'...")
        
        # Extract audio from video
        audio_path = self._extract_audio(video_path)
        audio_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        self.logger.info(f"Audio file size: {audio_size_mb:.2f} MB")
        
        try:
            # Prepare transcription options
            options = {
                "task": "transcribe",
                "verbose": True,  # Enable verbose output
                "fp16": False,    # Use FP32 for better compatibility
            }
            
            # Set language options
            if self.settings['detect_language']:
                self.logger.info("Using automatic language detection.")
            elif self.settings['languages']:
                language = self.settings['languages'][0]  # Use the first language in the list
                self.logger.info(f"Using specified language: {language}")
                options["language"] = language
            
            # Enable word-level timestamps for better segmentation
            if self.settings.get('word_timestamps', True):
                options["word_timestamps"] = True
                self.logger.info("Using word-level timestamps for better segmentation")
            
            # Log start of transcription
            self.logger.info(f"Starting Whisper transcription with model '{self.settings['model']}' on device '{self.settings['device']}'...")
            self.logger.info("This may take some time depending on the video length and model size...")
            
            # Perform transcription with timing
            self.logger.info("Loading audio into memory...")
            load_start = time.time()
            
            # Perform the transcription
            self.logger.info("Beginning transcription...")
            transcribe_start = time.time()
            result = self.model.transcribe(audio_path, **options)
            transcribe_end = time.time()
            
            # Log transcription timing
            transcribe_duration = transcribe_end - transcribe_start
            self.logger.info(f"Transcription processing took {transcribe_duration:.2f} seconds")
            
            # Create Transcription object
            transcription = Transcription(
                segments=result["segments"],
                language=result["language"]
            )
            
            # Log transcription details
            segment_count = len(transcription.get_segments())
            self.logger.info(f"Transcription completed with {segment_count} segments.")
            self.logger.info(f"Detected language: {transcription.get_language()}")
            
            # Log overall timing
            overall_duration = time.time() - overall_start_time
            self.logger.info(f"Total transcription process took {overall_duration:.2f} seconds")
            
            return transcription
        
        except Exception as e:
            self.logger.error(f"Error during transcription: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        finally:
            # Clean up temporary audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
                self.logger.info(f"Removed temporary audio file: {audio_path}")
    
    def save_srt(self, transcription: Transcription, output_path: str):
        """
        Save the transcription as an SRT file.
        
        Args:
            transcription: The transcription data.
            output_path: The path to save the SRT file.
        """
        self.logger.info(f"Saving SRT file to '{output_path}'...")
        
        try:
            # Get configuration settings
            max_line_width = self.settings.get('max_line_width', 20)
            max_line_count = self.settings.get('max_line_count', 1)
            sentence_aware = self.settings.get('sentence_aware', True)
            
            self.logger.info(f"Using settings: max_line_count={max_line_count}, max_line_width={max_line_width}, sentence_aware={sentence_aware}")
            
            # Get the segments from the transcription
            segments = transcription.get_segments()
            
            # Create subtitle entries from the segments
            subtitle_entries = self._create_subtitle_entries(
                segments, 
                max_line_width=max_line_width,
                sentence_aware=sentence_aware
            )
            
            # Write the SRT file directly
            self._write_srt_file(subtitle_entries, output_path)
            
            self.logger.info(f"SRT file saved successfully with {len(subtitle_entries)} entries.")
        
        except Exception as e:
            self.logger.error(f"Failed to save SRT file: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _create_subtitle_entries(self, segments: List[Dict[str, Any]], max_line_width: int, sentence_aware: bool) -> List[Dict[str, Any]]:
        """
        Create subtitle entries from segments.
        
        Args:
            segments: The segments from Whisper.
            max_line_width: Maximum line width.
            sentence_aware: Whether to use sentence-aware segmentation.
        
        Returns:
            A list of subtitle entries.
        """
        self.logger.info(f"Creating subtitle entries from {len(segments)} segments...")
        
        # Extract all words with their timestamps
        all_words = []
        for segment in segments:
            if 'words' in segment:
                all_words.extend(segment['words'])
        
        # If no word-level timestamps, use segment-level timestamps
        if not all_words and segments:
            self.logger.info("No word-level timestamps found. Using segment-level timestamps.")
            return self._process_segments(segments, max_line_width, sentence_aware)
        
        self.logger.info(f"Found {len(all_words)} words with timestamps")
        
        # Group words into subtitle entries
        entries = []
        current_entry = {
            'start': None,
            'end': None,
            'text': '',
            'words': []
        }
        
        current_length = 0
        sentence_endings = ['.', '!', '?']
        
        for word in all_words:
            word_text = word.get('word', '').strip()
            if not word_text:
                continue
            
            # Check if adding this word would exceed the max line width
            new_length = current_length + len(word_text) + (1 if current_length > 0 else 0)
            
            # Start a new entry if:
            # 1. This is the first word
            # 2. Adding this word would exceed max_line_width
            # 3. Sentence-aware is enabled and the previous word ends with a sentence ending
            start_new_entry = (
                current_entry['start'] is None or
                new_length > max_line_width or
                (sentence_aware and 
                 current_entry['text'] and 
                 any(current_entry['text'].endswith(ending) for ending in sentence_endings))
            )
            
            if start_new_entry and current_entry['start'] is not None:
                # Finalize the current entry
                entries.append(current_entry)
                
                # Start a new entry
                current_entry = {
                    'start': word.get('start', 0),
                    'end': word.get('end', 0),
                    'text': word_text,
                    'words': [word]
                }
                current_length = len(word_text)
            else:
                # Add to the current entry
                if current_entry['start'] is None:
                    current_entry['start'] = word.get('start', 0)
                
                current_entry['end'] = word.get('end', 0)
                
                # Add a space if this isn't the first word
                if current_entry['text']:
                    current_entry['text'] += ' '
                    current_length += 1
                
                current_entry['text'] += word_text
                current_entry['words'].append(word)
                current_length += len(word_text)
        
        # Add the last entry if it has content
        if current_entry['start'] is not None and current_entry['text']:
            entries.append(current_entry)
        
        # Validate and fix timestamps
        entries = self._validate_timestamps(entries)
        
        # Fix overlapping entries
        entries = self._fix_overlapping_entries(entries)
        
        self.logger.info(f"Created {len(entries)} subtitle entries")
        return entries
    
    def _process_segments(self, segments: List[Dict[str, Any]], max_line_width: int, sentence_aware: bool) -> List[Dict[str, Any]]:
        """
        Process segments when word-level timestamps are not available.
        
        Args:
            segments: The segments from Whisper.
            max_line_width: Maximum line width.
            sentence_aware: Whether to use sentence-aware segmentation.
        
        Returns:
            A list of processed segments.
        """
        self.logger.info("Processing segments without word-level timestamps...")
        
        # First, apply sentence-aware segmentation if enabled
        if sentence_aware:
            processed_segments = self._process_segments_sentence_aware(segments, max_line_width)
        else:
            processed_segments = segments.copy()
        
        # Validate timestamps
        validated_segments = self._validate_and_fix_timestamps(processed_segments)
        
        # Remove duplicates and fix overlaps
        final_segments = self._remove_duplicates_and_overlaps(validated_segments)
        
        return final_segments
    
    def _validate_timestamps(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate and fix timestamps in entries.
        
        Args:
            entries: The entries to validate.
        
        Returns:
            The validated entries.
        """
        self.logger.info("Validating timestamps...")
        
        validated_entries = []
        
        for entry in entries:
            # Skip empty entries
            if not entry.get('text', '').strip():
                continue
            
            # Ensure start and end times are valid
            if 'start' in entry and 'end' in entry:
                start = entry['start']
                end = entry['end']
                
                # Fix invalid timestamps (end before or equal to start)
                if end <= start:
                    # Set a minimum duration of 0.5 seconds
                    entry['end'] = start + 0.5
                    self.logger.warning(f"Fixed invalid timestamp: {start} -> {entry['end']} for text: '{entry['text']}'")
            
            validated_entries.append(entry)
        
        self.logger.info(f"Validated {len(entries)} entries, kept {len(validated_entries)} valid entries")
        return validated_entries
    
    def _fix_overlapping_entries(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fix overlapping entries.
        
        Args:
            entries: The entries to fix.
        
        Returns:
            The fixed entries.
        """
        self.logger.info("Fixing overlapping entries...")
        
        # Sort entries by start time
        sorted_entries = sorted(entries, key=lambda e: e.get('start', 0))
        
        # Fix overlaps
        for i in range(1, len(sorted_entries)):
            if sorted_entries[i]['start'] < sorted_entries[i-1]['end']:
                # Set start time to the end time of the previous entry
                sorted_entries[i]['start'] = sorted_entries[i-1]['end'] + 0.01
                
                # Ensure end time is after start time
                if sorted_entries[i]['end'] <= sorted_entries[i]['start']:
                    sorted_entries[i]['end'] = sorted_entries[i]['start'] + 0.5
                
                self.logger.info(f"Fixed overlapping entry: '{sorted_entries[i]['text']}' now at {sorted_entries[i]['start']} -> {sorted_entries[i]['end']}")
        
        return sorted_entries
    
    def _write_srt_file(self, entries: List[Dict[str, Any]], output_path: str):
        """
        Write entries to an SRT file.
        
        Args:
            entries: The entries to write.
            output_path: The path to the output file.
        """
        self.logger.info(f"Writing {len(entries)} entries to SRT file: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, entry in enumerate(entries, 1):
                # Format timestamps
                start_time = self._format_timestamp(entry['start'])
                end_time = self._format_timestamp(entry['end'])
                
                # Write the entry
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{entry['text']}\n\n")
    
    def _format_timestamp(self, seconds: float) -> str:
        """
        Format seconds into SRT timestamp.
        
        Args:
            seconds: The time in seconds.
        
        Returns:
            The formatted timestamp.
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        whole_seconds = int(seconds)
        milliseconds = int((seconds - whole_seconds) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"
    
    def _validate_and_fix_timestamps(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate and fix timestamps in segments.
        
        Args:
            segments: The segments to validate.
        
        Returns:
            The validated segments.
        """
        self.logger.info("Validating and fixing timestamps...")
        
        validated_segments = []
        
        for segment in segments:
            # Skip empty segments
            if not segment.get('text', '').strip():
                continue
            
            # Ensure start and end times are valid
            if 'start' in segment and 'end' in segment:
                start = segment['start']
                end = segment['end']
                
                # Fix invalid timestamps (end before start)
                if end <= start:
                    # Set a minimum duration of 0.5 seconds
                    segment['end'] = start + 0.5
                    self.logger.warning(f"Fixed invalid timestamp: {start} -> {segment['end']} for text: '{segment['text']}'")
            
            validated_segments.append(segment)
        
        self.logger.info(f"Validated {len(segments)} segments, kept {len(validated_segments)} valid segments")
        return validated_segments
    
    def _remove_duplicates_and_overlaps(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate and overlapping segments.
        
        Args:
            segments: The segments to process.
        
        Returns:
            The processed segments.
        """
        self.logger.info("Removing duplicates and fixing overlapping segments...")
        
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda s: s.get('start', 0))
        
        # Remove duplicates and fix overlaps
        unique_segments = []
        seen_texts = set()
        last_end_time = 0
        
        for segment in sorted_segments:
            text = segment.get('text', '').strip()
            
            # Skip empty segments
            if not text:
                continue
            
            # Skip duplicate text
            if text in seen_texts:
                self.logger.info(f"Skipping duplicate text: '{text}'")
                continue
            
            # Fix overlapping timestamps
            if 'start' in segment and segment['start'] < last_end_time:
                # Set start time to the end time of the previous segment
                segment['start'] = last_end_time
                
                # Ensure end time is after start time
                if segment['end'] <= segment['start']:
                    segment['end'] = segment['start'] + 0.5
                
                self.logger.info(f"Fixed overlapping segment: '{text}' now at {segment['start']} -> {segment['end']}")
            
            # Add to unique segments
            unique_segments.append(segment)
            seen_texts.add(text)
            
            # Update last end time
            if 'end' in segment:
                last_end_time = segment['end']
        
        self.logger.info(f"Removed duplicates and fixed overlaps: {len(segments)} -> {len(unique_segments)} segments")
        return unique_segments
    
    def _process_segments_sentence_aware(self, segments: List[Dict[str, Any]], max_line_width: int) -> List[Dict[str, Any]]:
        """
        Process segments to break at sentence boundaries when possible.
        
        Args:
            segments: The original segments.
            max_line_width: The maximum line width.
        
        Returns:
            The processed segments.
        """
        self.logger.info("Processing segments with sentence-aware segmentation...")
        
        # Define sentence-ending punctuation
        sentence_endings = ['.', '!', '?']
        
        # Create a copy of the segments to avoid modifying the original
        processed_segments = []
        
        for segment in segments:
            text = segment.get('text', '')
            
            # Skip empty segments
            if not text.strip():
                continue
            
            # If the text is already shorter than the max line width, keep it as is
            if len(text) <= max_line_width:
                processed_segments.append(segment.copy())
                continue
            
            # Look for sentence boundaries within the max line width
            found_boundary = False
            for i in range(min(max_line_width, len(text) - 1), 0, -1):
                if text[i] in sentence_endings and (i == len(text) - 1 or text[i + 1].isspace()):
                    # Found a sentence boundary
                    found_boundary = True
                    
                    # Create two segments: one up to the boundary, one after
                    segment1 = segment.copy()
                    segment1['text'] = text[:i + 1].strip()
                    
                    segment2 = segment.copy()
                    segment2['text'] = text[i + 1:].strip()
                    
                    # Adjust the start and end times for the second segment
                    # This is a simplification; ideally we would use word timestamps
                    if 'start' in segment and 'end' in segment:
                        duration = segment['end'] - segment['start']
                        ratio = len(segment1['text']) / len(text)
                        segment1['end'] = segment['start'] + duration * ratio
                        segment2['start'] = segment1['end']
                    
                    # Add the first segment
                    processed_segments.append(segment1)
                    
                    # Process the second segment recursively if it's still too long
                    if len(segment2['text']) > max_line_width:
                        processed_segments.extend(
                            self._process_segments_sentence_aware([segment2], max_line_width)
                        )
                    else:
                        processed_segments.append(segment2)
                    
                    break
            
            # If no sentence boundary was found, use the original segment
            if not found_boundary:
                processed_segments.append(segment.copy())
        
        self.logger.info(f"Processed {len(segments)} segments into {len(processed_segments)} segments")
        return processed_segments
    
    def _extract_audio(self, video_path: str) -> str:
        """
        Extract audio from a video file.
        
        Args:
            video_path: The path to the video file.
        
        Returns:
            The path to the extracted audio file.
        """
        import time
        start_time = time.time()
        
        # Get file size for logging
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        self.logger.info(f"Extracting audio from video ({file_size_mb:.2f} MB)...")
        
        # Create a temporary file for the audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            audio_path = temp_file.name
        
        # First attempt with simplified command
        try:
            # Simplified FFmpeg command - more direct approach
            command = [
                "ffmpeg",
                "-i", video_path,
                "-vn",                # No video
                "-acodec", "pcm_s16le", # PCM 16-bit audio
                "-ar", "16000",       # 16kHz sample rate (good for speech)
                "-ac", "1",           # Mono audio
                "-y",                 # Overwrite output file
                audio_path
            ]
            
            self.logger.info(f"Executing FFmpeg command (attempt 1): {' '.join(command)}")
            
            # Run the command with direct output to console
            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120  # 2-minute timeout
            )
            
            # Check if the process was successful and the file has content
            if process.returncode == 0 and os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                elapsed_time = time.time() - start_time
                self.logger.info(f"Audio extracted successfully in {elapsed_time:.2f} seconds.")
                self.logger.info(f"Audio file size: {os.path.getsize(audio_path) / 1024:.2f} KB")
                return audio_path
            
            # If we get here, the first attempt failed or produced an empty file
            self.logger.warning("First extraction attempt failed or produced empty file. Trying alternative method...")
            
            if process.stderr:
                self.logger.warning(f"FFmpeg stderr: {process.stderr}")
        
        except subprocess.TimeoutExpired:
            self.logger.warning("First extraction attempt timed out. Trying alternative method...")
        
        except Exception as e:
            self.logger.warning(f"First extraction attempt failed: {str(e)}. Trying alternative method...")
        
        # Second attempt with alternative command
        try:
            # Alternative FFmpeg command
            command = [
                "ffmpeg",
                "-i", video_path,
                "-vn",                # No video
                "-f", "wav",          # Force WAV format
                "-y",                 # Overwrite output file
                audio_path
            ]
            
            self.logger.info(f"Executing FFmpeg command (attempt 2): {' '.join(command)}")
            
            # Run the command with direct output to console
            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120  # 2-minute timeout
            )
            
            # Check if the process was successful
            if process.returncode != 0:
                self.logger.error(f"FFmpeg process failed with return code {process.returncode}")
                if process.stderr:
                    self.logger.error(f"FFmpeg error output: {process.stderr}")
                raise RuntimeError("Failed to extract audio with both methods")
            
            # Check if the output file exists and has content
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                self.logger.error("FFmpeg completed but no audio file was created")
                raise RuntimeError("Failed to extract audio: No output file created")
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Audio extracted successfully with alternative method in {elapsed_time:.2f} seconds.")
            self.logger.info(f"Audio file size: {os.path.getsize(audio_path) / 1024:.2f} KB")
            
            return audio_path
        
        except subprocess.TimeoutExpired:
            self.logger.error("FFmpeg process timed out after 2 minutes")
            if 'process' in locals():
                process.kill()
            if os.path.exists(audio_path):
                os.remove(audio_path)
            raise RuntimeError("Audio extraction timed out. The video might be too large or FFmpeg encountered an issue.")
        
        except Exception as e:
            self.logger.error(f"Failed to extract audio with both methods: {str(e)}")
            
            # Clean up temporary file
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            raise RuntimeError(f"Failed to extract audio: {str(e)}")
