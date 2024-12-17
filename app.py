import json
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError
import openai  # Placeholder for GPT-4 integration
from src.common import DataExtractionHandler
import ollama
from src.logger import logging

# Define the main extraction handler class
class ModelHandler:
    def __init__(self, model: str = "gpt-4"):
        self.model = model
        logging.info(f"Initialized DataExtractionHandler with model: {self.model}")

    def extract_with_gpt4(self, prompt: str) -> Optional[dict]:
        """
        Extracts data using GPT-4 models based on the provided prompt.

        Args:
            prompt (str): The input prompt.

        Returns:
            dict: Extracted data or None if the model fails to extract.
        """
        logging.debug(f"Sending prompt to {self.model}: {prompt}")
        try:
            # Replace with actual GPT-4 API call
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data extraction assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            logging.info("Data successfully extracted from GPT-4.")
            return json.loads(response["choices"][0]["message"]["content"])
        except Exception as e:
            logging.error(f"Error during GPT-4 extraction: {e}")
            return None
        
    def extract_with_llama(self, prompt: str) -> Optional[dict]:
        """
            Extracts data using Llama models based on the provided prompt.

            Args:
                prompt (str): The input prompt.

            Returns:
                dict: Extracted data or None if the model fails to extract.
        """
        try:
            logging.debug(f"Sending prompt to {self.model}: {prompt}")
            stream = ollama.chat(
            model='llama',
            messages=[{'role': 'user', 'content': 'Name an engineer that passes the vibe check'}],
            stream=True
            ) 
            logging.info("Data successfully extracted from LLAMA.")
            for chunk in stream:
                print(chunk['message']['content'], end='', flush=True)
        except Exception as e:
            logging.error(f"Error during GPT-4 extraction: {e}")
            return None

# Example usage
if __name__ == "__main__":
    model_handler = ModelHandler()
    handler = DataExtractionHandler()

    # Generate prompt for GPT-4
    prompt = "Extract the following fields based on the schema: ... (provide details from PDF)"
    # gpt4_result = model_handler.extract_with_gpt4(prompt)
    gpt4_result = model_handler.extract_with_llama(prompt)

    if gpt4_result:
        logging.debug("GPT-4 Extraction Result:")
        logging.debug(gpt4_result)

        # Validate and save the result
        validation_result = handler.validate_data(gpt4_result)
        logging.debug(validation_result)

        # Save validated JSON
        handler.save_json_to_file(validation_result, "output.json")
    else:
        logging.error("Failed to extract data using GPT-4.")
