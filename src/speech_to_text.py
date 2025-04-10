import os
import uuid
import time # Import time for potential delays/polling if needed, though result() handles it
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import storage # Import GCS client
from google.api_core.exceptions import GoogleAPICallError, NotFound
from typing import Optional, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeechToText:
    # Added bucket_name to init
    def __init__(self, bucket_name: Optional[str] = None): 
        """
        Initialize the Speech-to-Text service.

        Args:
            bucket_name (Optional[str]): The GCS bucket name for long audio uploads. 
                                         Required if using transcribe_audio for >60s audio.
        """
        try:
            self.speech_client = speech.SpeechClient()
            # Initialize storage client only if a bucket name is provided
            self.storage_client = storage.Client() if bucket_name else None
            self.bucket_name = bucket_name
        except Exception as e:
            logger.exception("Failed to initialize Google Cloud clients. Check credentials.")
            # Raise a more specific error or handle appropriately
            raise RuntimeError("Could not initialize Google Cloud clients") from e

        # Configuration (keep previous settings, adjust if needed)
        self.encoding = speech.RecognitionConfig.AudioEncoding.WEBM_OPUS 
        self.sample_rate_hertz = 48000 
        self.language_code = "en-US"
        
        # Create custom vocabulary (remains the same)
        self.custom_vocabulary = self._create_custom_vocabulary()

    def _create_custom_vocabulary(self) -> Dict[str, list]:
        """
        Create a dictionary of custom vocabularies for different technical domains.
        """
        # (Vocabulary definition remains the same as before)
        return {
            "data_science": [
                "algorithm", "neural network", "regression", "classification",
                "machine learning", "deep learning", "tensorflow", "pytorch"
            ],
            "software_engineering": [
                "framework", "architecture", "design pattern", "api",
                "database", "optimization", "performance"
            ],
            "devops": [
                "docker", "kubernetes", "ci/cd", "infrastructure",
                "automation", "monitoring", "logging"
            ]
            # Add other domains if needed
        }

    def _upload_to_gcs(self, audio_bytes: bytes) -> Optional[str]:
        """
        Uploads audio bytes to the configured GCS bucket.

        Args:
            audio_bytes: The audio data as bytes.

        Returns:
            The gs:// URI of the uploaded file, or None if upload fails.
        """
        if not self.storage_client or not self.bucket_name:
            logger.error("GCS client or bucket name not configured. Cannot upload.")
            return None
            
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            # Generate a unique name for the audio file
            blob_name = f"audio-uploads/{uuid.uuid4()}.opus" # Assuming opus format
            blob = bucket.blob(blob_name)

            logger.info(f"Uploading audio to gs://{self.bucket_name}/{blob_name}")
            # Upload data. Consider content_type if needed, but often inferred.
            blob.upload_from_string(audio_bytes) 
            
            gcs_uri = f"gs://{self.bucket_name}/{blob_name}"
            logger.info(f"Successfully uploaded to {gcs_uri}")
            return gcs_uri

        except NotFound:
            logger.error(f"GCS bucket '{self.bucket_name}' not found.")
            return None
        except Exception as e:
            logger.exception(f"Failed to upload audio to GCS bucket '{self.bucket_name}': {e}")
            return None

    def transcribe_audio(self, audio_content: bytes, domain: str = "general") -> Optional[str]:
        """
        Transcribe audio content to text using LongRunningRecognize for potentially long audio.
        
        Args:
            audio_content: The audio content (bytes) to transcribe.
            domain: The technical domain for custom vocabulary.
            
        Returns:
            The transcribed text or None if transcription fails.
        """
        gcs_uri = None
        try:
            # 1. Upload audio to GCS
            gcs_uri = self._upload_to_gcs(audio_content)
            if not gcs_uri:
                # Error logged in _upload_to_gcs
                raise ValueError("Failed to upload audio to GCS.")

            # 2. Configure the recognition request
            config = speech.RecognitionConfig(
                encoding=self.encoding, # Should be WEBM_OPUS
                sample_rate_hertz=self.sample_rate_hertz, # Should be 48000
                language_code=self.language_code,
                enable_automatic_punctuation=True,
                model="latest_long" # Use model suitable for long audio
                # Consider adding audio_channel_count=1 if mono
            )

            # Add custom vocabulary if domain is specified
            # (Logic remains the same, but commented out previously - restore if needed)
            # if domain != "general" and domain in self.custom_vocabulary:
            #     config.speech_contexts = [
            #         speech.SpeechContext(phrases=self.custom_vocabulary[domain])
            #     ]

            # 3. Create audio request using the GCS URI
            audio = speech.RecognitionAudio(uri=gcs_uri)

            # 4. Perform the long-running transcription request
            logger.info(f"Starting long-running transcription for {gcs_uri}")
            operation = self.speech_client.long_running_recognize(config=config, audio=audio)

            # 5. Wait for the operation to complete
            # Set a timeout appropriate for expected max audio length (e.g., 360s for ~5-6 mins)
            timeout_seconds = 360 
            logger.info(f"Waiting for transcription operation to complete (timeout: {timeout_seconds}s)...")
            response = operation.result(timeout=timeout_seconds) 
            logger.info("Transcription operation completed.")

            # 6. Process the response
            final_transcript = ""
            highest_confidence = 0.0
            if hasattr(response, 'results') and response.results:
                # Concatenate transcripts from all results
                for result in response.results:
                     if hasattr(result, 'alternatives') and result.alternatives:
                         # Get the alternative with the highest confidence
                         best_alternative = result.alternatives[0]
                         final_transcript += best_alternative.transcript + " "
                         highest_confidence = max(highest_confidence, best_alternative.confidence)
            
            final_transcript = final_transcript.strip() # Remove trailing space

            if not final_transcript:
                 logger.warning(f"Transcription result was empty for {gcs_uri}.")
                 return None # Or return "" depending on desired behavior

            logger.info(f"Transcription confidence (highest): {highest_confidence:.2f}")
            logger.info(f"Transcription result for {gcs_uri}: {final_transcript[:100]}...") # Log snippet
            return final_transcript

        except GoogleAPICallError as e:
            # Log the full error for more details
            logger.error(f"Speech-to-Text API error occurred: {e}") 
            return None
        except Exception as e:
            # Catch other potential errors (timeout, upload failure, etc.)
            logger.exception(f"Unexpected error during transcription process: {e}")
            return None
        finally:
            # Optional: Clean up the GCS file after processing
            if gcs_uri:
                try:
                    # Extract bucket and blob name from URI
                    if gcs_uri.startswith("gs://"):
                        parts = gcs_uri[5:].split('/', 1)
                        if len(parts) == 2:
                            bucket_name, blob_name = parts
                            bucket = self.storage_client.bucket(bucket_name)
                            blob = bucket.blob(blob_name)
                            if blob.exists():
                                logger.info(f"Deleting temporary GCS file: {gcs_uri}")
                                blob.delete()
                except Exception as cleanup_error:
                    logger.warning(f"Failed to delete temporary GCS file {gcs_uri}: {cleanup_error}")


    # stream_transcribe method remains the same as before...
    def stream_transcribe(self, audio_generator, domain: str = "general") -> str:
        """
        Stream audio content and transcribe in real-time.
        
        Args:
            audio_generator: Generator that yields audio chunks
            domain: The technical domain for custom vocabulary
            
        Returns:
            The final transcribed text
        """
        config = speech.RecognitionConfig(
            encoding=self.encoding, # Use WEBM_OPUS
            sample_rate_hertz=self.sample_rate_hertz, # Use 48000
            language_code=self.language_code,
            enable_automatic_punctuation=True,
            model="latest_short" # Streaming often uses short model
        )

        # if domain != "general" and domain in self.custom_vocabulary:
        #     config.speech_contexts = [
        #         speech.SpeechContext(
        #             phrases=self.custom_vocabulary[domain]
        #         )
        #     ]

        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=True
        )

        requests = (
            speech.StreamingRecognizeRequest(audio_content=chunk)
            for chunk in audio_generator
        )

        try:
            responses = self.speech_client.streaming_recognize(
                streaming_config, requests
            )

            final_transcript = ""
            for response in responses:
                if not response.results:
                    continue

                result = response.results[0]
                if not result.alternatives:
                    continue

                transcript = result.alternatives[0].transcript
                confidence = result.alternatives[0].confidence

                if result.is_final:
                    final_transcript = transcript
                    logger.info(f"Final transcript: {transcript}")
                    logger.info(f"Confidence: {confidence:.2f}")

            return final_transcript

        except Exception as e:
            logger.error(f"Streaming transcription error: {str(e)}")
            return ""

# Example usage (remains the same, but needs bucket name)
def main():
    # !!! IMPORTANT: Replace with your actual bucket name !!!
    BUCKET_FOR_TESTING = "your-gcs-bucket-name-here" 
    if BUCKET_FOR_TESTING == "your-gcs-bucket-name-here":
         print("Please update BUCKET_FOR_TESTING in the main() function of speech_to_text.py")
         return

    stt = SpeechToText(bucket_name=BUCKET_FOR_TESTING)
    
    # Example audio content (replace with actual audio bytes from a file > 60s)
    # e.g., with open("long_audio.opus", "rb") as f: audio_content = f.read()
    audio_content = b'...' # Placeholder - needs real long audio bytes
    
    if len(audio_content) <= 3:
         print("Placeholder audio_content is too short. Replace with real audio bytes.")
         return

    # Transcribe audio
    transcript = stt.transcribe_audio(audio_content, domain="software_engineering")
    print(f"Transcribed text: {transcript}")

if __name__ == "__main__":
    main()
