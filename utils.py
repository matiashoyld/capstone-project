import os
from openai import OpenAI
from google import genai
from google.genai import types

def get_r1_response(prompt):
    """
    Get a response from the DeepSeek API using the R1 model.
    
    Args:
        prompt (str): The user prompt to send to the model
        include_reasoning (bool, optional): Whether to include the model's reasoning in the response.
                                           Defaults to False.
    
    Returns:
        tuple: A tuple containing the model's response text and reasoning (if requested)
    """
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
    
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": prompt}]
    )
    
    content =  response.choices[0].message.content
    reasoning = response.choices[0].message.reasoning_content
    
    return content, reasoning

def get_flash_response(prompt):
    """
    Get a response from the Gemini API using the flash-thinking model.
    
    Args:
        prompt (str): The user prompt to send to the model
    
    Returns:
        tuple: A tuple containing the model's thinking process and final response
    """
    
    # Modify the prompt to instruct the model to use thinking tags
    modified_prompt = f"""
    ### Begin Meta Instructions ###
    For the following task, first think step by step through the problem using <thinking></thinking> tags. 
    After your thinking process, provide your final response after the </thinking> tag.
    ### End Meta Instructions ###

    Here's the task:
    #### Begin Task ####

    {prompt}

    #### End Task ####
    """
    
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )
    
    model = "gemini-2.0-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=modified_prompt),
            ],
        ),
    ]
    
    generate_content_config = types.GenerateContentConfig(
        temperature=0.7,
        top_p=0.95,
        top_k=64,
        max_output_tokens=65536,
        response_mime_type="text/plain",
    )
    
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    
    # Extract the full response text
    full_text = response.candidates[0].content.parts[0].text
    
    # Extract reasoning and content based on thinking tags
    if "<thinking>" in full_text and "</thinking>" in full_text:
        # Extract the text between <thinking> and </thinking> tags
        thinking_start = full_text.find("<thinking>") + len("<thinking>")
        thinking_end = full_text.find("</thinking>")
        reasoning = full_text[thinking_start:thinking_end].strip()
        
        # Extract the content after the </thinking> tag
        content = full_text[thinking_end + len("</thinking>"):].strip()
    else:
        # If no thinking tags are found, return the full text as content
        reasoning = ""
        content = full_text
    
    return content, reasoning
