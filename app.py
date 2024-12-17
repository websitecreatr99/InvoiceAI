import logging
import json
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError
import openai  # Placeholder for GPT-4 integration
from helper.common import DataExtractionHandler
import ollama

# Configure logger
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the main extraction handler class
class ModelHandler:
    def __init__(self, model: str = "gpt-4"):
        self.model = model
        logger.info(f"Initialized DataExtractionHandler with model: {self.model}")

    def extract_with_gpt4(self, prompt: str) -> Optional[dict]:
        """
        Extracts data using GPT-4 models based on the provided prompt.

        Args:
            prompt (str): The input prompt.

        Returns:
            dict: Extracted data or None if the model fails to extract.
        """
        logger.debug(f"Sending prompt to {self.model}: {prompt}")
        try:
            # Replace with actual GPT-4 API call
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data extraction assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            logger.info("Data successfully extracted from GPT-4.")
            return json.loads(response["choices"][0]["message"]["content"])
        except Exception as e:
            logger.error(f"Error during GPT-4 extraction: {e}")
            return None
        
    def extract_with_ollama(self, prompt: str) -> Optional[dict]:
        """
            Extracts data using Llama models based on the provided prompt.

            Args:
                prompt (str): The input prompt.

            Returns:
                dict: Extracted data or None if the model fails to extract.
        """
        try:
            logger.debug(f"Sending prompt to {self.model}: {prompt}")
            stream = ollama.chat(
            model='mistral',
            messages=[{'role': 'user', 'content': 'Name an engineer that passes the vibe check'}],
            stream=True
            ) 
            logger.info("Data successfully extracted from LLAMA.")
            for chunk in stream:
                print(chunk['message']['content'], end='', flush=True)
        except Exception as e:
            logger.error(f"Error during GPT-4 extraction: {e}")
            return None

# Example usage
if __name__ == "__main__":
    model_handler = ModelHandler()
    handler = DataExtractionHandler()

    # Generate prompt for GPT-4
    prompt = "Extract the following fields based on the schema: ... (provide details from PDF)"
    gpt4_result = model_handler.extract_with_gpt4(prompt)

    if gpt4_result:
        logger.debug("GPT-4 Extraction Result:")
        logger.debug(gpt4_result)

        # Validate and save the result
        validation_result = handler.validate_data(gpt4_result)
        logger.debug(validation_result)

        # Save validated JSON
        handler.save_json_to_file(validation_result, "output.json")
    else:
        logger.error("Failed to extract data using GPT-4.")
