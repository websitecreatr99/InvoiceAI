from ..app import logger
from pydantic import BaseModel, Field, ValidationError

class DataExtractionHandler:

    def validate_data(self, data: dict) -> str:
        """
        Validates and parses input data into JSON format based on the defined schema.

        Args:
            data (dict): Input data to be validated and parsed.

        Returns:
            str: Validated data in JSON format or error message in case of validation failure.
        """
        logger.debug("Validating extracted data against the schema.")
        try:
            parsed_data = Extraction(**data)
            logger.info("Data validation successful.")
            return parsed_data.json(indent=4)
        except ValidationError as e:
            logger.error(f"Validation Error: {e}")
            return f"Validation Error: {e}"

    def save_json_to_file(self, json_data: str, filename: str):
        """
        Saves the JSON data to a file.

        Args:
            json_data (str): JSON string to save.
            filename (str): Name of the file to save the JSON data.
        """
        logger.debug(f"Saving JSON data to file: {filename}")
        try:
            with open(filename, 'w') as file:
                file.write(json_data)
            logger.info(f"JSON data successfully saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving JSON data to file: {e}")