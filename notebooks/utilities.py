import re
import ast
import yaml
import os
import json
from openai import OpenAI
from typing import Dict, Any, Optional

def extract_options(option_text: str) -> list:
    """
    Extract and clean options from a string containing either JSON-like list or YAML-formatted strings.
    
    Args:
        option_text (str): Input string containing options in JSON or YAML format
        
    Returns:
        list: List of cleaned option strings. Returns list of 5 None values if parsing fails.
        
    Raises:
        ValueError: If input string cannot be parsed as JSON or YAML
        SyntaxError: If input string has invalid syntax
    """
    try:
        # Handle JSON-like list
        if option_text.startswith('['):
            try:
                options_list = ast.literal_eval(option_text)
                # Remove extra quotes from each element
                options_list = [elem.strip('"') for elem in options_list]
                return options_list
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing JSON-like list: {e}")
                return [None] * 5

        # Handle YAML-like formatted strings 
        else:
            try:
                # First, we need to clean up the string by removing the outer quotes and replacing \\n with \n
                cleaned = option_text.strip('"').replace('\\n', '\n')

                # Parse with yaml
                parsed_list = yaml.safe_load(cleaned)

                # If it's a list, clean up each element
                if isinstance(parsed_list, list):
                    cleaned_list = []
                    for elem in parsed_list:
                        if isinstance(elem, str):
                            # Remove extra quotes and unnecessary escape sequences
                            cleaned_elem = elem.replace('\\"', '"').strip('"')

                            # Replace multiple backslashes with single backslash
                            cleaned_elem = re.sub(r'\\\\+', r'\\', cleaned_elem)
                            
                            # Remove any leading or trailing quotes
                            cleaned_elem = cleaned_elem.strip('"')
                            cleaned_list.append(cleaned_elem)
                        else:
                            cleaned_list.append(elem)
                    return cleaned_list
            except yaml.YAMLError as e:
                print(f"Error parsing YAML string: {e}")
                return [None] * 5

    except Exception as e:
        print(f"Unexpected error: {e}")
        return [None] * 5

    # Fallback: Return five None values if parsing fails
    return [None] * 5

def get_openai_response(
    prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    response_format: Optional[dict] = None,
    api_key: Optional[str] = None
) -> Optional[str]:
    """
    Call OpenAI API and get the model's response.
    
    Args:
        prompt (str): The input prompt for the model
        model (str): The OpenAI model to use (default: "gpt-3.5-turbo")
        temperature (float): Controls randomness in the output (0.0-1.0)
        response_format (Optional[dict]): Format specification for the response. Use {"type": "json_object"} to get JSON response
        api_key (Optional[str]): OpenAI API key. If None, will try to get from environment
        
    Returns:
        str: The model's response text
    """
    try:
        # Get API key from parameter or environment variable
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not provided and not found in environment variables")

        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)

        # Prepare API call parameters
        params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature
        }
        
        # Add response format if specified
        if response_format:
            params["response_format"] = response_format

        # Make the API call
        response = client.chat.completions.create(**params)

        # Extract the response content
        return response.choices[0].message.content

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None

def parse_openai_response(response_text: str) -> Optional[dict]:
    """
    Parse the JSON response from OpenAI into a structured format.
    
    Args:
        response_text (str): The raw JSON response text from OpenAI
        
    Returns:
        Optional[dict]: Dictionary containing parsed response with keys:
            - thinking: str, the thinking process explanation
            - answer: str, the selected answer (a-e)
            - confidence: int, confidence score (0-100)
            Returns None if parsing fails
    """
    try:
        # Parse JSON response
        response_json = json.loads(response_text)
        
        # Extract fields from nested response structure
        thinking = response_json["response"]["thinking"]
        answer = response_json["response"]["answer"].strip().lower()
        confidence = int(response_json["response"]["confidence"])
        
        # Validate answer is a single letter a-e
        if not answer in ['a', 'b', 'c', 'd', 'e']:
            raise ValueError(f"Invalid answer format: {answer}")
            
        # Validate confidence is 0-100
        if not 0 <= confidence <= 100:
            raise ValueError(f"Confidence must be between 0-100, got: {confidence}")
            
        return {
            "thinking": thinking,
            "answer": answer,
            "confidence": confidence
        }
        
    except Exception as e:
        print(f"Error parsing OpenAI response: {e}")
        return None
