import json
import os
import re
import base64
from io import BytesIO

import requests
# from retry import retry  # Comment out if not available

from PIL import Image
import numpy as np
import torch

# from .session import get_client  # Comment out if not available

MAX_RETRY = 3

CLAUDE3_MAX_SIZE = 1568

# Mock bedrock client for testing - replace with actual client
try:
    from .session import get_client
    bedrock_runtime_client = get_client(service_name="bedrock-runtime")
    bedrock_client = get_client(service_name="bedrock")
except:
    # Mock client for testing
    class MockClient:
        def invoke_model(self, **kwargs):
            return {"body": type('MockBody', (), {'read': lambda: '{"content": [{"text": "mock response"}]}'})()}
        def list_inference_profiles(self, **kwargs):
            return {"inferenceProfileSummaries": []}
    bedrock_runtime_client = MockClient()
    bedrock_client = MockClient()

# Mock retry decorator if not available
try:
    from retry import retry
except ImportError:
    def retry(tries=MAX_RETRY):
        def decorator(func):
            return func
        return decorator

# Global cache for inference profiles
_inference_profiles_cache = None

def get_inference_profiles():
    """
    Retrieve and cache inference profiles from AWS Bedrock.
    Returns a dictionary mapping model IDs to their inference profile ARNs.
    """
    global _inference_profiles_cache
    
    if _inference_profiles_cache is not None:
        return _inference_profiles_cache
    
    try:
        response = bedrock_client.list_inference_profiles()
        profiles = {}
        
        for profile in response.get('inferenceProfileSummaries', []):
            profile_arn = profile.get('inferenceProfileArn')
            profile_id = profile.get('inferenceProfileId')
            
            # Extract model ID from profile ID (e.g., us.anthropic.claude-3-5-sonnet-20240620-v1:0 -> anthropic.claude-3-5-sonnet-20240620-v1:0)
            if profile_id and profile_arn:
                # Remove the region prefix (us., eu., etc.)
                model_id = '.'.join(profile_id.split('.')[1:])
                profiles[model_id] = profile_arn
        
        _inference_profiles_cache = profiles
        print(f"Loaded {len(profiles)} inference profiles")
        return profiles
        
    except Exception as e:
        print(f"Error: Could not retrieve inference profiles: {e}")
        raise RuntimeError(f"Failed to retrieve inference profiles from AWS Bedrock: {e}")

def get_inference_profile_arn(model_id):
    """
    Get the inference profile ARN for a given model ID.
    
    Args:
        model_id (str): The model ID to look up
        
    Returns:
        str: The inference profile ARN
        
    Raises:
        ValueError: If model ID not found in available profiles
    """
    profiles = get_inference_profiles()
    arn = profiles.get(model_id)
    
    if not arn:
        available_models = list(profiles.keys())
        raise ValueError(f"Model '{model_id}' not found in available inference profiles. Available models: {available_models}")
    
    return arn

def get_available_inference_profile_arns(model_ids):
    """
    Get available inference profile ARNs for a list of model IDs.
    
    Args:
        model_ids (list): List of model IDs to look up
        
    Returns:
        list: List of available inference profile ARNs
        
    Raises:
        ValueError: If no models found in available profiles
    """
    profiles = get_inference_profiles()
    available_arns = []
    
    for model_id in model_ids:
        arn = profiles.get(model_id)
        if arn:
            available_arns.append(arn)
    
    if not available_arns:
        available_models = list(profiles.keys())
        raise ValueError(f"None of the requested models {model_ids} found in available inference profiles. Available models: {available_models}")
    
    return available_arns



class BedrockNovaMultimodal:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model_id": (
                    [
                        "amazon.nova-premier-v1:0",
                        "amazon.nova-pro-v1:0",
                        "amazon.nova-lite-v1:0",
                        "amazon.nova-micro-v1:0",
                    ],
                ),
                "maxTokens": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 0,  # Minimum value
                        "max": 4096,  # Maximum value
                        "step": 1,  # Slider's step
                        "display": "number",  # Cosmetic only: display as "number" or "slider"
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "slider",
                    },
                ),
                "topP": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "slider",
                    },
                )
            },
            "optional": {
                "image": ("IMAGE",{"default": None}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "forward"
    CATEGORY = "aws"

    @retry(tries=MAX_RETRY)
    def forward(
        self,
        **kwargs
    ):
        
        prompt = kwargs.get('prompt')
        model_id = kwargs.get('model_id')
        maxTokens = kwargs.get('maxTokens')
        temperature = kwargs.get('temperature')
        topP = kwargs.get('topP')
        image = kwargs.get('image')
        
        content = []
        content.append({
                            "text": prompt,
                        })

        if image is not None:
            image = image[0] * 255.0
            image = Image.fromarray(image.clamp(0, 255).numpy().round().astype(np.uint8))
    
            width, height = image.size
            max_size = max(width, height)
            if max_size > CLAUDE3_MAX_SIZE:
                width = round(width * CLAUDE3_MAX_SIZE / max_size)
                height = round(height * CLAUDE3_MAX_SIZE / max_size)
                image = image.resize((width, height))
    
            buffer = BytesIO()
            image.save(buffer, format="png", quality=80)
            image_data = buffer.getvalue()
    
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            content.append( {
                "image": {
                    "format": "png",
                    "source": {"bytes": image_base64},
                }
              }
            )
            
        inf_params = {"max_new_tokens": maxTokens, "top_p": topP, "temperature": temperature}


        body = json.dumps(
            {
                "schemaVersion": "messages-v1",
                "messages": [
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                "inferenceConfig": inf_params,
            },
            ensure_ascii=False,
        )

        response = bedrock_runtime_client.invoke_model(
            body=body,
            modelId=model_id,
        )

        message = json.loads(response.get("body").read())['output']['message']['content'][0]['text']
        print("here1===",message)
        return (message,)

class BedrockClaudeMultimodal:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "model_id": (
                    [
                        # Latest multimodal Claude models (ACTIVE)
                        "anthropic.claude-3-5-sonnet-20240620-v1:0",
                        "anthropic.claude-3-5-sonnet-20241022-v2:0",
                        "anthropic.claude-3-7-sonnet-20250219-v1:0",
                        "anthropic.claude-opus-4-20250514-v1:0",
                        "anthropic.claude-sonnet-4-20250514-v1:0",
                        "anthropic.claude-3-haiku-20240307-v1:0",
                        "anthropic.claude-3-sonnet-20240229-v1:0",
                        "anthropic.claude-3-opus-20240229-v1:0",
                    ],
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 0,  # Minimum value
                        "max": 4096,  # Maximum value
                        "step": 1,  # Slider's step
                        "display": "number",  # Cosmetic only: display as "number" or "slider"
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "slider",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "slider",
                    },
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 250,
                        "min": 0,
                        "max": 500,
                        "step": 1,
                        "round": 1,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "slider",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "forward"
    CATEGORY = "aws"

    @retry(tries=MAX_RETRY)
    def forward(
        self,
        image,
        prompt,
        model_id,
        max_tokens,
        temperature,
        top_p,
        top_k,
    ):
        """
        Invokes the Anthropic Claude model to run an inference using the input
        provided in the request body.

        :param prompt: The prompt that you want Claude to complete.
        :param inference_profile: Optional inference profile for Claude 4 models.
        :return: Inference response from the model.
        """

        image = image[0] * 255.0
        image = Image.fromarray(image.clamp(0, 255).numpy().round().astype(np.uint8))

        width, height = image.size
        max_size = max(width, height)
        if max_size > CLAUDE3_MAX_SIZE:
            width = round(width * CLAUDE3_MAX_SIZE / max_size)
            height = round(height * CLAUDE3_MAX_SIZE / max_size)
            image = image.resize((width, height))

        buffer = BytesIO()
        image.save(buffer, format="webp", quality=80)
        image_data = buffer.getvalue()

        image_base64 = base64.b64encode(image_data).decode("utf-8")

        # The different model providers have individual request and response formats.
        # For the format, ranges, and default values for Anthropic Claude, refer to:
        # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html

        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/webp",
                                    "data": image_base64,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            },
            ensure_ascii=False,
        )

        response = bedrock_runtime_client.invoke_model(
            body=body,
            modelId=model_id,
        )
        message = json.loads(response.get("body").read())["content"][0]["text"]

        return (message,)


class BedrockClaude:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model_id": (
                    [
                        # Latest Claude models (ACTIVE)
                        "anthropic.claude-3-5-sonnet-20240620-v1:0",
                        "anthropic.claude-3-5-sonnet-20241022-v2:0",
                        "anthropic.claude-3-5-haiku-20241022-v1:0",
                        "anthropic.claude-3-7-sonnet-20250219-v1:0",
                        "anthropic.claude-opus-4-20250514-v1:0",
                        "anthropic.claude-sonnet-4-20250514-v1:0",
                        "anthropic.claude-3-haiku-20240307-v1:0",
                        "anthropic.claude-3-sonnet-20240229-v1:0",
                        "anthropic.claude-3-opus-20240229-v1:0",
                        # Legacy models (still available)
                        "anthropic.claude-v2:1",
                        "anthropic.claude-v2",
                        "anthropic.claude-instant-v1",
                    ],
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 0,  # Minimum value
                        "max": 4096,  # Maximum value
                        "step": 1,  # Slider's step
                        "display": "number",  # Cosmetic only: display as "number" or "slider"
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "slider",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "slider",
                    },
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 250,
                        "min": 0,
                        "max": 500,
                        "step": 1,
                        "round": 1,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "slider",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    CATEGORY = "aws"

    @retry(tries=MAX_RETRY)
    def forward(
        self,
        prompt,
        model_id,
        max_tokens,
        temperature,
        top_p,
        top_k,
    ):
        """
        Invokes the Anthropic Claude model to run an inference using the input
        provided in the request body.

        :param prompt: The prompt that you want Claude to complete.
        :param inference_profile: Optional inference profile for Claude 4 models.
        :return: Inference response from the model.
        """

        # The different model providers have individual request and response formats.
        # For the format, ranges, and default values for Anthropic Claude, refer to:
        # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html
        print("prompt input:",prompt)
        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            }
                        ],
                    }
                ],
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            },
            ensure_ascii=False,
        )

        response = bedrock_runtime_client.invoke_model(
            body=body,
            modelId=model_id,
        )
        message = json.loads(response.get("body").read())["content"][0]["text"]
        print("output message:",message)

        return (message,)




class BedrockClaude4:
    @classmethod
    def INPUT_TYPES(s):
        # Get available Claude 4 inference profile ARNs dynamically
        claude4_models = [
            "anthropic.claude-3-7-sonnet-20250219-v1:0",
            "anthropic.claude-opus-4-20250514-v1:0", 
            "anthropic.claude-sonnet-4-20250514-v1:0"
        ]
        available_arns = get_available_inference_profile_arns(claude4_models)
        
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "inference_profile_arn": (available_arns,),
                "max_tokens": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 0,  # Minimum value
                        "max": 4096,  # Maximum value
                        "step": 1,  # Slider's step
                        "display": "number",  # Cosmetic only: display as "number" or "slider"
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "slider",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "slider",
                    },
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 250,
                        "min": 0,
                        "max": 500,
                        "step": 1,
                        "round": 1,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "slider",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "forward"
    CATEGORY = "aws"

    @retry(tries=MAX_RETRY)
    def forward(
        self,
        prompt,
        inference_profile_arn,
        max_tokens,
        temperature,
        top_p,
        top_k,
    ):
        """
        Invokes the Anthropic Claude 4 model to run an inference using the input
        provided in the request body.

        :param prompt: The prompt that you want Claude to complete.
        :param inference_profile_arn: The ARN of the inference profile to use.
        :return: Inference response from the model.
        """

        # The different model providers have individual request and response formats.
        # For the format, ranges, and default values for Anthropic Claude, refer to:
        # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html
        print("prompt input:", prompt)
        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            }
                        ],
                    }
                ],
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            },
            ensure_ascii=False,
        )

        # For Claude 4, we use the inference profile ARN as the modelId
        response = bedrock_runtime_client.invoke_model(
            body=body,
            modelId=inference_profile_arn,
        )
        message = json.loads(response.get("body").read())["content"][0]["text"]
        print("output message:", message)

        return (message,)


class BedrockClaude4Multimodal:
    @classmethod
    def INPUT_TYPES(s):
        # Get available Claude 4 multimodal inference profile ARNs dynamically
        claude4_multimodal_models = [
            "anthropic.claude-3-7-sonnet-20250219-v1:0",
            "anthropic.claude-opus-4-20250514-v1:0", 
            "anthropic.claude-sonnet-4-20250514-v1:0"
        ]
        available_arns = get_available_inference_profile_arns(claude4_multimodal_models)
        
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "inference_profile_arn": (available_arns,),
                "max_tokens": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 0,  # Minimum value
                        "max": 4096,  # Maximum value
                        "step": 1,  # Slider's step
                        "display": "number",  # Cosmetic only: display as "number" or "slider"
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "slider",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "slider",
                    },
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 250,
                        "min": 0,
                        "max": 500,
                        "step": 1,
                        "round": 1,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                        "display": "slider",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "forward"
    CATEGORY = "aws"

    @retry(tries=MAX_RETRY)
    def forward(
        self,
        image,
        prompt,
        inference_profile_arn,
        max_tokens,
        temperature,
        top_p,
        top_k,
    ):
        """
        Invokes the Anthropic Claude 4 model to run a multimodal inference using the input
        provided in the request body.

        :param image: The image to analyze.
        :param prompt: The prompt that you want Claude to complete.
        :param inference_profile_arn: The ARN of the inference profile to use.
        :return: Inference response from the model.
        """

        image = image[0] * 255.0
        image = Image.fromarray(image.clamp(0, 255).numpy().round().astype(np.uint8))

        width, height = image.size
        max_size = max(width, height)
        if max_size > CLAUDE3_MAX_SIZE:
            width = round(width * CLAUDE3_MAX_SIZE / max_size)
            height = round(height * CLAUDE3_MAX_SIZE / max_size)
            image = image.resize((width, height))

        buffer = BytesIO()
        image.save(buffer, format="webp", quality=80)
        image_data = buffer.getvalue()

        image_base64 = base64.b64encode(image_data).decode("utf-8")

        # The different model providers have individual request and response formats.
        # For the format, ranges, and default values for Anthropic Claude, refer to:
        # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html

        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/webp",
                                    "data": image_base64,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            },
            ensure_ascii=False,
        )

        # For Claude 4, we use the inference profile ARN as the modelId
        response = bedrock_runtime_client.invoke_model(
            body=body,
            modelId=inference_profile_arn,
        )
        message = json.loads(response.get("body").read())["content"][0]["text"]

        return (message,)


class BedrockLlama4:
    @classmethod
    def INPUT_TYPES(s):
        # Get available inference profile ARNs dynamically
        llama4_models = [
            "meta.llama4-scout-17b-instruct-v1:0",
            "meta.llama4-maverick-17b-instruct-v1:0"
        ]
        available_arns = get_available_inference_profile_arns(llama4_models)
        default_arn = available_arns[0]  # Use first available ARN as default
        
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "inference_profile_arn": ("STRING", {"multiline": False, "default": default_arn}),
                "max_tokens": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,
                        "display": "slider",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,
                        "display": "slider",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "forward"
    CATEGORY = "aws"

    @retry(tries=MAX_RETRY)
    def forward(
        self,
        prompt,
        inference_profile_arn,
        max_tokens,
        temperature,
        top_p,
    ):
        """
        Invokes the Meta Llama 4 model to run an inference using the input
        provided in the request body.

        :param prompt: The prompt that you want Llama to complete.
        :param inference_profile_arn: The ARN of the inference profile to use.
        :return: Inference response from the model.
        """
        print("prompt input:", prompt)
        
        # Llama 4 uses a different request format than Claude
        # Note: system_prompt is not supported, removed from request
        body = json.dumps(
            {
                "prompt": prompt,
                "max_gen_len": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
            ensure_ascii=False,
        )

        # Use inference profile ARN for Llama 4
        response = bedrock_runtime_client.invoke_model(
            body=body,
            modelId=inference_profile_arn,
        )
        
        # Parse the response based on Llama 4's response format
        response_body = json.loads(response.get("body").read())
        message = response_body.get("generation", "")
        print("output message:", message)

        return (message,)


class BedrockLlama4Multimodal:
    @classmethod
    def INPUT_TYPES(s):
        # Get available inference profile ARNs dynamically
        llama4_models = [
            "meta.llama4-scout-17b-instruct-v1:0",
            "meta.llama4-maverick-17b-instruct-v1:0"
        ]
        available_arns = get_available_inference_profile_arns(llama4_models)
        default_arn = available_arns[0]  # Use first available ARN as default
        
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "inference_profile_arn": ("STRING", {"multiline": False, "default": default_arn}),
                "max_tokens": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,
                        "display": "slider",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,
                        "display": "slider",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "forward"
    CATEGORY = "aws"

    @retry(tries=MAX_RETRY)
    def forward(
        self,
        image,
        prompt,
        inference_profile_arn,
        max_tokens,
        temperature,
        top_p,
    ):
        """
        Invokes the Meta Llama 4 model to run a multimodal inference using the input
        provided in the request body.

        :param image: The image to analyze.
        :param prompt: The prompt that you want Llama to complete.
        :param inference_profile_arn: The ARN of the inference profile to use.
        :return: Inference response from the model.
        """
        # Process the image
        image = image[0] * 255.0
        image = Image.fromarray(image.clamp(0, 255).numpy().round().astype(np.uint8))

        # Resize if needed (Llama 4 may have different size requirements than Claude)
        max_size = 2048  # Adjust based on Llama 4's requirements
        width, height = image.size
        if max(width, height) > max_size:
            scale = max_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height))

        # Convert to base64
        buffer = BytesIO()
        image.save(buffer, format="jpeg", quality=95)
        image_data = buffer.getvalue()
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        # Note: Multimodal format for Llama 4 may need adjustment
        # Current format may not be correct - needs testing
        body = json.dumps(
            {
                "prompt": prompt,
                "max_gen_len": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                # Image format may need to be different for Llama 4
                "image": {
                    "format": "jpeg",
                    "data": image_base64
                }
            },
            ensure_ascii=False,
        )

        # Use inference profile ARN for Llama 4
        response = bedrock_runtime_client.invoke_model(
            body=body,
            modelId=inference_profile_arn,
        )
        
        # Parse the response based on Llama 4's response format
        response_body = json.loads(response.get("body").read())
        message = response_body.get("generation", "")

        return (message,)



class BedrockDeepSeekR1:
    @classmethod
    def INPUT_TYPES(s):
        # Get available inference profile ARN dynamically
        deepseek_model = "deepseek.r1-v1:0"
        arn = get_inference_profile_arn(deepseek_model)
        
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "inference_profile_arn": ("STRING", {"multiline": False, "default": arn}),
                "max_tokens": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,
                        "display": "slider",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,
                        "display": "slider",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "forward"
    CATEGORY = "aws"

    @retry(tries=MAX_RETRY)
    def forward(
        self,
        prompt,
        inference_profile_arn,
        max_tokens,
        temperature,
        top_p,
    ):
        """
        Invokes the DeepSeek R1 model to run an inference using the input
        provided in the request body.

        :param prompt: The prompt that you want DeepSeek to complete.
        :param inference_profile_arn: The ARN of the inference profile to use.
        :return: Inference response from the model.
        """
        print("prompt input:", prompt)
        
        # DeepSeek R1 uses OpenAI-compatible format
        body = json.dumps(
            {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
            ensure_ascii=False,
        )

        # Use inference profile ARN for DeepSeek R1
        response = bedrock_runtime_client.invoke_model(
            body=body,
            modelId=inference_profile_arn,
        )
        
        # Parse the response based on DeepSeek R1's response format
        response_body = json.loads(response.get("body").read())
        message = response_body['choices'][0]['message']['content']
        print("output message:", message)

        return (message,)


class BedrockLlama3:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model_id": (
                    [
                        "meta.llama3-8b-instruct-v1:0",
                        "meta.llama3-70b-instruct-v1:0",
                    ],
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,
                        "display": "slider",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,
                        "display": "slider",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "forward"
    CATEGORY = "aws"

    @retry(tries=MAX_RETRY)
    def forward(
        self,
        prompt,
        model_id,
        max_tokens,
        temperature,
        top_p,
    ):
        """
        Invokes the Meta Llama 3 model to run an inference using the input
        provided in the request body.

        :param prompt: The prompt that you want Llama to complete.
        :return: Inference response from the model.
        """
        print("prompt input:", prompt)
        
        # Llama 3 uses a different request format than Claude
        # Note: system_prompt is not supported for on-demand Llama models
        body = json.dumps(
            {
                "prompt": prompt,
                "max_gen_len": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
            ensure_ascii=False,
        )

        response = bedrock_runtime_client.invoke_model(
            body=body,
            modelId=model_id,
        )
        
        # Parse the response based on Llama's response format
        response_body = json.loads(response.get("body").read())
        message = response_body.get("generation", "")
        print("output message:", message)

        return (message,)


class BedrockLlama31:
    @classmethod
    def INPUT_TYPES(s):
        # Get available inference profile ARNs dynamically
        llama31_models = [
            "meta.llama3-1-8b-instruct-v1:0",
            "meta.llama3-1-70b-instruct-v1:0"
        ]
        available_arns = get_available_inference_profile_arns(llama31_models)
        
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "inference_profile_arn": (available_arns,),
                "max_tokens": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,
                        "display": "slider",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,
                        "display": "slider",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "forward"
    CATEGORY = "aws"

    @retry(tries=MAX_RETRY)
    def forward(
        self,
        prompt,
        inference_profile_arn,
        max_tokens,
        temperature,
        top_p,
    ):
        """
        Invokes the Meta Llama 3.1 model to run an inference using the input
        provided in the request body.

        :param prompt: The prompt that you want Llama to complete.
        :param inference_profile_arn: The ARN of the inference profile to use.
        :return: Inference response from the model.
        """
        print("prompt input:", prompt)
        
        # Llama 3.1 uses a different request format than Claude
        # Note: system_prompt is not supported
        body = json.dumps(
            {
                "prompt": prompt,
                "max_gen_len": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
            ensure_ascii=False,
        )

        # Use inference profile ARN for Llama 3.1
        response = bedrock_runtime_client.invoke_model(
            body=body,
            modelId=inference_profile_arn,
        )
        
        # Parse the response based on Llama's response format
        response_body = json.loads(response.get("body").read())
        message = response_body.get("generation", "")
        print("output message:", message)

        return (message,)


class BedrockAI21Jamba:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model_id": (
                    [
                        "ai21.jamba-1-5-large-v1:0",
                        "ai21.jamba-1-5-mini-v1:0",
                        "ai21.jamba-instruct-v1:0",  # Legacy but still available
                    ],
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,
                        "display": "slider",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,
                        "display": "slider",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "forward"
    CATEGORY = "aws"

    @retry(tries=MAX_RETRY)
    def forward(
        self,
        prompt,
        model_id,
        max_tokens,
        temperature,
        top_p,
    ):
        """
        Invokes the AI21 Jamba model to run an inference using the input
        provided in the request body.

        :param prompt: The prompt that you want Jamba to complete.
        :return: Inference response from the model.
        """
        print("prompt input:", prompt)
        
        # AI21 Jamba uses a specific format
        body = json.dumps(
            {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
            ensure_ascii=False,
        )

        response = bedrock_runtime_client.invoke_model(
            body=body,
            modelId=model_id,
        )
        
        # Parse the response based on AI21's response format
        response_body = json.loads(response.get("body").read())
        message = response_body['choices'][0]['message']['content']
        print("output message:", message)

        return (message,)


class BedrockCohere:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model_id": (
                    [
                        "cohere.command-r-plus-v1:0",
                        "cohere.command-r-v1:0",
                        "cohere.command-text-v14",
                        "cohere.command-light-text-v14",
                    ],
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,
                        "display": "slider",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,
                        "display": "slider",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "forward"
    CATEGORY = "aws"

    @retry(tries=MAX_RETRY)
    def forward(
        self,
        prompt,
        model_id,
        max_tokens,
        temperature,
        top_p,
    ):
        """
        Invokes the Cohere model to run an inference using the input
        provided in the request body.

        :param prompt: The prompt that you want Cohere to complete.
        :return: Inference response from the model.
        """
        print("prompt input:", prompt)
        
        # Cohere uses a specific format
        body = json.dumps(
            {
                "message": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "p": top_p,
            },
            ensure_ascii=False,
        )

        response = bedrock_runtime_client.invoke_model(
            body=body,
            modelId=model_id,
        )
        
        # Parse the response based on Cohere's response format
        response_body = json.loads(response.get("body").read())
        message = response_body['text']
        print("output message:", message)

        return (message,)


class BedrockTitan:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model_id": (
                    [
                        "amazon.titan-text-premier-v1:0",
                        "amazon.titan-text-express-v1",
                        "amazon.titan-text-lite-v1",
                        "amazon.titan-tg1-large",
                    ],
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,
                        "display": "slider",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,
                        "display": "slider",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "forward"
    CATEGORY = "aws"

    @retry(tries=MAX_RETRY)
    def forward(
        self,
        prompt,
        model_id,
        max_tokens,
        temperature,
        top_p,
    ):
        """
        Invokes the Amazon Titan model to run an inference using the input
        provided in the request body.

        :param prompt: The prompt that you want Titan to complete.
        :return: Inference response from the model.
        """
        print("prompt input:", prompt)
        
        # Titan uses a specific format
        body = json.dumps(
            {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": max_tokens,
                    "temperature": temperature,
                    "topP": top_p,
                }
            },
            ensure_ascii=False,
        )

        response = bedrock_runtime_client.invoke_model(
            body=body,
            modelId=model_id,
        )
        
        # Parse the response based on Titan's response format
        response_body = json.loads(response.get("body").read())
        message = response_body['results'][0]['outputText']
        print("output message:", message)

        return (message,)


class BedrockLlama32Multimodal:
    @classmethod
    def INPUT_TYPES(s):
        # Get available inference profile ARNs dynamically (only multimodal capable models)
        llama32_multimodal_models = [
            "meta.llama3-2-11b-instruct-v1:0",
            "meta.llama3-2-90b-instruct-v1:0"
        ]
        available_arns = get_available_inference_profile_arns(llama32_multimodal_models)
        
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "inference_profile_arn": (available_arns,),
                "max_tokens": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,
                        "display": "slider",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,
                        "display": "slider",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "forward"
    CATEGORY = "aws"

    @retry(tries=MAX_RETRY)
    def forward(
        self,
        image,
        prompt,
        inference_profile_arn,
        max_tokens,
        temperature,
        top_p,
    ):
        """
        Invokes the Meta Llama 3.2 multimodal model to run an inference using the input
        provided in the request body.

        :param image: The image to analyze.
        :param prompt: The prompt that you want Llama to complete.
        :param inference_profile_arn: The ARN of the inference profile to use.
        :return: Inference response from the model.
        """
        # Process the image
        image = image[0] * 255.0
        image = Image.fromarray(image.clamp(0, 255).numpy().round().astype(np.uint8))

        # Resize if needed
        max_size = 2048
        width, height = image.size
        if max(width, height) > max_size:
            scale = max_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height))

        # Convert to base64
        buffer = BytesIO()
        image.save(buffer, format="jpeg", quality=95)
        image_data = buffer.getvalue()
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        # Llama 3.2 multimodal format (may need adjustment)
        body = json.dumps(
            {
                "prompt": prompt,
                "max_gen_len": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "image": {
                    "format": "jpeg",
                    "data": image_base64
                }
            },
            ensure_ascii=False,
        )

        # Use inference profile ARN for Llama 3.2
        response = bedrock_runtime_client.invoke_model(
            body=body,
            modelId=inference_profile_arn,
        )
        
        # Parse the response based on Llama's response format
        response_body = json.loads(response.get("body").read())
        message = response_body.get("generation", "")

        return (message,)


class BedrockLlama32:
    @classmethod
    def INPUT_TYPES(s):
        # Get available inference profile ARNs dynamically
        llama32_models = [
            "meta.llama3-2-1b-instruct-v1:0",
            "meta.llama3-2-3b-instruct-v1:0",
            "meta.llama3-2-11b-instruct-v1:0",
            "meta.llama3-2-90b-instruct-v1:0"
        ]
        available_arns = get_available_inference_profile_arns(llama32_models)
        
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "inference_profile_arn": (available_arns,),
                "max_tokens": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,
                        "display": "slider",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,
                        "display": "slider",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "forward"
    CATEGORY = "aws"

    @retry(tries=MAX_RETRY)
    def forward(
        self,
        prompt,
        inference_profile_arn,
        max_tokens,
        temperature,
        top_p,
    ):
        """
        Invokes the Meta Llama 3.2 model to run an inference using the input
        provided in the request body.

        :param prompt: The prompt that you want Llama to complete.
        :param inference_profile_arn: The ARN of the inference profile to use.
        :return: Inference response from the model.
        """
        print("prompt input:", prompt)
        
        # Llama 3.2 uses a different request format than Claude
        # Note: system_prompt is not supported
        body = json.dumps(
            {
                "prompt": prompt,
                "max_gen_len": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
            ensure_ascii=False,
        )

        # Use inference profile ARN for Llama 3.2
        response = bedrock_runtime_client.invoke_model(
            body=body,
            modelId=inference_profile_arn,
        )
        
        # Parse the response based on Llama's response format
        response_body = json.loads(response.get("body").read())
        message = response_body.get("generation", "")
        print("output message:", message)

        return (message,)


class BedrockLlama33:
    @classmethod
    def INPUT_TYPES(s):
        # Get available inference profile ARN dynamically
        llama33_model = "meta.llama3-3-70b-instruct-v1:0"
        arn = get_inference_profile_arn(llama33_model)
        
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "inference_profile_arn": ("STRING", {"multiline": False, "default": arn}),
                "max_tokens": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,
                        "display": "slider",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,
                        "display": "slider",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "forward"
    CATEGORY = "aws"

    @retry(tries=MAX_RETRY)
    def forward(
        self,
        prompt,
        inference_profile_arn,
        max_tokens,
        temperature,
        top_p,
    ):
        """
        Invokes the Meta Llama 3.3 model to run an inference using the input
        provided in the request body.

        :param prompt: The prompt that you want Llama to complete.
        :param inference_profile_arn: The ARN of the inference profile to use.
        :return: Inference response from the model.
        """
        print("prompt input:", prompt)
        
        # Llama 3.3 uses a different request format than Claude
        # Note: system_prompt is not supported
        body = json.dumps(
            {
                "prompt": prompt,
                "max_gen_len": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
            ensure_ascii=False,
        )

        # Use inference profile ARN for Llama 3.3
        response = bedrock_runtime_client.invoke_model(
            body=body,
            modelId=inference_profile_arn,
        )
        
        # Parse the response based on Llama's response format
        response_body = json.loads(response.get("body").read())
        message = response_body.get("generation", "")
        print("output message:", message)

        return (message,)


class BedrockMistral:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model_id": (
                    [
                        "mistral.mistral-7b-instruct-v0:2",
                        "mistral.mixtral-8x7b-instruct-v0:1",
                        "mistral.mistral-large-2402-v1:0",
                        "mistral.mistral-small-2402-v1:0",
                    ],
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,
                        "display": "slider",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,
                        "display": "slider",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "forward"
    CATEGORY = "aws"

    @retry(tries=MAX_RETRY)
    def forward(
        self,
        prompt,
        model_id,
        max_tokens,
        temperature,
        top_p,
    ):
        """
        Invokes the Mistral model to run an inference using the input
        provided in the request body.

        :param prompt: The prompt that you want Mistral to complete.
        :return: Inference response from the model.
        """
        print("prompt input:", prompt)
        
        # Mistral uses a specific format for AWS Bedrock
        body = json.dumps(
            {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
            ensure_ascii=False,
        )

        response = bedrock_runtime_client.invoke_model(
            body=body,
            modelId=model_id,
        )
        
        # Parse the response based on Mistral's response format
        response_body = json.loads(response.get("body").read())
        message = response_body['outputs'][0]['text']
        print("output message:", message)

        return (message,)


class BedrockPixtralLarge:
    @classmethod
    def INPUT_TYPES(s):
        # Get available inference profile ARN dynamically
        pixtral_model = "mistral.pixtral-large-2502-v1:0"
        arn = get_inference_profile_arn(pixtral_model)
        
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "inference_profile_arn": ("STRING", {"multiline": False, "default": arn}),
                "max_tokens": (
                    "INT",
                    {
                        "default": 2048,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,
                        "display": "slider",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,
                        "display": "slider",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "forward"
    CATEGORY = "aws"

    @retry(tries=MAX_RETRY)
    def forward(
        self,
        image,
        prompt,
        inference_profile_arn,
        max_tokens,
        temperature,
        top_p,
    ):
        """
        Invokes the Mistral Pixtral Large model to run a multimodal inference using the input
        provided in the request body.

        :param image: The image to analyze.
        :param prompt: The prompt that you want Pixtral to complete.
        :param inference_profile_arn: The ARN of the inference profile to use.
        :return: Inference response from the model.
        """
        # Process the image
        image = image[0] * 255.0
        image = Image.fromarray(image.clamp(0, 255).numpy().round().astype(np.uint8))

        # Resize if needed
        max_size = 2048
        width, height = image.size
        if max(width, height) > max_size:
            scale = max_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height))

        # Convert to base64
        buffer = BytesIO()
        image.save(buffer, format="jpeg", quality=95)
        image_data = buffer.getvalue()
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        # Pixtral Large multimodal format (similar to OpenAI)
        body = json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
            ensure_ascii=False,
        )

        # Use inference profile ARN for Pixtral Large
        response = bedrock_runtime_client.invoke_model(
            body=body,
            modelId=inference_profile_arn,
        )
        
        # Parse the response based on Pixtral's response format
        response_body = json.loads(response.get("body").read())
        message = response_body['choices'][0]['message']['content']

        return (message,)


NODE_CLASS_MAPPINGS = {
    # Claude Models (TEXT output)
    "Bedrock - Claude": BedrockClaude,
    "Bedrock - Claude Multimodal": BedrockClaudeMultimodal,
    "Bedrock - Claude 4": BedrockClaude4,
    "Bedrock - Claude 4 Multimodal": BedrockClaude4Multimodal,
    
    # Llama Models (TEXT output)
    "Bedrock - Llama 3": BedrockLlama3,
    "Bedrock - Llama 3.1": BedrockLlama31,
    "Bedrock - Llama 3.2": BedrockLlama32,
    "Bedrock - Llama 3.2 Multimodal": BedrockLlama32Multimodal,
    "Bedrock - Llama 3.3": BedrockLlama33,
    "Bedrock - Llama 4": BedrockLlama4,
    "Bedrock - Llama 4 Multimodal": BedrockLlama4Multimodal,
    
    # Other Text Generation Models
    "Bedrock - DeepSeek R1": BedrockDeepSeekR1,
    "Bedrock - Mistral": BedrockMistral,
    "Bedrock - Pixtral Large": BedrockPixtralLarge,
    "Bedrock - AI21 Jamba": BedrockAI21Jamba,
    "Bedrock - Cohere": BedrockCohere,
    "Bedrock - Titan": BedrockTitan,
    "Bedrock - Nova": BedrockNovaMultimodal
}
