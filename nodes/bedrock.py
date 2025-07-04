"""
Fixed version of as.py with proper error handling for ComfyUI environment
This version won't crash when AWS credentials are not available
"""

import json
import os
import re
import base64
from io import BytesIO

import requests
from retry import retry

MAX_RETRY = 3

CLAUDE3_MAX_SIZE = 1568

# Mock bedrock client for testing - replace with actual client
from .session import get_client
bedrock_runtime_client = get_client(service_name="bedrock-runtime")
bedrock_client = get_client(service_name="bedrock")

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
        print(f"Warning: Could not retrieve inference profiles: {e}")
        
        # Provide specific guidance for awsvpc mode
        error_msg = str(e).lower()
        if 'credentials' in error_msg or 'access' in error_msg or 'unauthorized' in error_msg:
            print("\n🔧 AWS Credential Configuration Needed:")
            print("For awsvpc mode (ECS/EKS environments):")
            print("  • ECS Tasks: Attach IAM Task Role with bedrock:* permissions")
            print("  • EKS Pods: Use IAM Roles for Service Accounts (IRSA)")
            print("  • Set AWS_DEFAULT_REGION environment variable")
            print("  • Ensure the role has these permissions:")
            print("    - bedrock:InvokeModel")
            print("    - bedrock:ListInferenceProfiles")
            print("    - bedrock:GetInferenceProfile")
        elif 'region' in error_msg:
            print("\n🌍 Region Configuration Needed:")
            print("Set AWS_DEFAULT_REGION environment variable, e.g.:")
            print("  export AWS_DEFAULT_REGION=us-east-1")
        
        print("\nThe nodes will use manual ARN input until credentials are configured.")
        _inference_profiles_cache = {}
        return {}

def get_inference_profile_arn_safe(model_id):
    """
    Safely get the inference profile ARN for a given model ID.
    Returns None if not found instead of raising an error.
    
    Args:
        model_id (str): The model ID to look up
        
    Returns:
        str or None: The inference profile ARN, or None if not found
    """
    profiles = get_inference_profiles()
    return profiles.get(model_id)

def get_available_inference_profile_arns_safe(model_ids):
    """
    Safely get available inference profile ARNs for a list of model IDs.
    Returns empty list if none found instead of raising an error.
    
    Args:
        model_ids (list): List of model IDs to look up
        
    Returns:
        list: List of available inference profile ARNs (may be empty)
    """
    profiles = get_inference_profiles()
    available_arns = []
    
    for model_id in model_ids:
        arn = profiles.get(model_id)
        if arn:
            available_arns.append(arn)
    
    return available_arns

# Keep the original functions for backward compatibility, but make them safer
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
    
    if not profiles:
        # If no profiles available, return a placeholder that indicates the issue
        print(f"Warning: No inference profiles available. AWS credentials may not be configured.")
        print(f"Returning placeholder ARN for {model_id}")
        return f"aws-credentials-not-configured-{model_id}"
    
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
    
    if not profiles:
        # If no profiles available, return placeholders that indicate the issue
        print(f"Warning: No inference profiles available. AWS credentials may not be configured.")
        print(f"Returning placeholder ARNs for {model_ids}")
        return [f"aws-credentials-not-configured-{model_id}" for model_id in model_ids]
    
    available_arns = []
    
    for model_id in model_ids:
        arn = profiles.get(model_id)
        if arn:
            available_arns.append(arn)
    
    if not available_arns:
        available_models = list(profiles.keys())
        raise ValueError(f"None of the requested models {model_ids} found in available inference profiles. Available models: {available_models}")
    
    return available_arns


# Now define the classes with better error handling

class BedrockClaude4:
    @classmethod
    def INPUT_TYPES(s):
        # Try to get available Claude 4 inference profile ARNs dynamically
        claude4_models = [
            "anthropic.claude-3-7-sonnet-20250219-v1:0",
            "anthropic.claude-opus-4-20250514-v1:0", 
            "anthropic.claude-sonnet-4-20250514-v1:0"
        ]
        
        # Use safe version that doesn't raise errors
        available_arns = get_available_inference_profile_arns_safe(claude4_models)
        
        if available_arns:
            # Dynamic dropdown if profiles are available
            arn_input = (available_arns,)
        else:
            # Manual text input if no profiles available
            print("Warning: No Claude 4 inference profiles found. Using manual ARN input.")
            arn_input = ("STRING", {"multiline": False, "default": "Enter Claude 4 inference profile ARN"})
        
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "inference_profile_arn": arn_input,
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
                        "default": 0.5,
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
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,
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
                        "round": 1,
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


# Add similar safe handling to other classes...
# (I'll show one more example and then provide the pattern)

class BedrockLlama4:
    @classmethod
    def INPUT_TYPES(s):
        # Try to get available Llama 4 inference profile ARNs dynamically
        llama4_models = [
            "meta.llama4-scout-17b-instruct-v1:0",
            "meta.llama4-maverick-17b-instruct-v1:0"
        ]
        
        # Use safe version that doesn't raise errors
        available_arns = get_available_inference_profile_arns_safe(llama4_models)
        
        if available_arns:
            # Use first available ARN as default for text input
            default_arn = available_arns[0]
        else:
            # Manual text input if no profiles available
            print("Warning: No Llama 4 inference profiles found. Using manual ARN input.")
            default_arn = "Enter Llama 4 inference profile ARN"
        
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


class BedrockClaude35Sonnet:
    @classmethod
    def INPUT_TYPES(s):
        # Try to get available Claude 3.5 Sonnet inference profile ARNs dynamically
        claude35_models = [
            "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "anthropic.claude-3-5-sonnet-20241022-v2:0"
        ]
        
        # Use safe version that doesn't raise errors
        available_arns = get_available_inference_profile_arns_safe(claude35_models)
        
        if available_arns:
            # Dynamic dropdown if profiles are available
            arn_input = (available_arns,)
        else:
            # Manual text input if no profiles available
            print("Warning: No Claude 3.5 Sonnet inference profiles found. Using manual ARN input.")
            arn_input = ("STRING", {"multiline": False, "default": "Enter Claude 3.5 Sonnet inference profile ARN"})
        
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "inference_profile_arn": arn_input,
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
                        "default": 0.5,
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
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,
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
                        "round": 1,
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
        Invokes the Anthropic Claude 3.5 Sonnet model to run an inference using the input
        provided in the request body.

        :param prompt: The prompt that you want Claude to complete.
        :param inference_profile_arn: The ARN of the inference profile to use.
        :return: Inference response from the model.
        """
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

        response = bedrock_runtime_client.invoke_model(
            body=body,
            modelId=inference_profile_arn,
        )
        message = json.loads(response.get("body").read())["content"][0]["text"]
        print("output message:", message)

        return (message,)


class BedrockClaude3Haiku:
    @classmethod
    def INPUT_TYPES(s):
        # Try to get available Claude 3 Haiku inference profile ARNs dynamically
        claude3_models = [
            "anthropic.claude-3-haiku-20240307-v1:0"
        ]
        
        # Use safe version that doesn't raise errors
        available_arns = get_available_inference_profile_arns_safe(claude3_models)
        
        if available_arns:
            # Dynamic dropdown if profiles are available
            arn_input = (available_arns,)
        else:
            # Manual text input if no profiles available
            print("Warning: No Claude 3 Haiku inference profiles found. Using manual ARN input.")
            arn_input = ("STRING", {"multiline": False, "default": "Enter Claude 3 Haiku inference profile ARN"})
        
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "inference_profile_arn": arn_input,
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
                        "default": 0.5,
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
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,
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
                        "round": 1,
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
        Invokes the Anthropic Claude 3 Haiku model to run an inference using the input
        provided in the request body.

        :param prompt: The prompt that you want Claude to complete.
        :param inference_profile_arn: The ARN of the inference profile to use.
        :return: Inference response from the model.
        """
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

        response = bedrock_runtime_client.invoke_model(
            body=body,
            modelId=inference_profile_arn,
        )
        message = json.loads(response.get("body").read())["content"][0]["text"]
        print("output message:", message)

        return (message,)


class BedrockMistralLarge:
    @classmethod
    def INPUT_TYPES(s):
        # Try to get available Mistral Large inference profile ARNs dynamically
        mistral_models = [
            "mistral.mistral-large-2402-v1:0",
            "mistral.mistral-large-2407-v1:0"
        ]
        
        # Use safe version that doesn't raise errors
        available_arns = get_available_inference_profile_arns_safe(mistral_models)
        
        if available_arns:
            # Use first available ARN as default for text input
            default_arn = available_arns[0]
        else:
            # Manual text input if no profiles available
            print("Warning: No Mistral Large inference profiles found. Using manual ARN input.")
            default_arn = "Enter Mistral Large inference profile ARN"
        
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
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,
                        "display": "slider",
                    },
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 50,
                        "min": 0,
                        "max": 200,
                        "step": 1,
                        "round": 1,
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
        Invokes the Mistral Large model to run an inference using the input
        provided in the request body.

        :param prompt: The prompt that you want Mistral to complete.
        :param inference_profile_arn: The ARN of the inference profile to use.
        :return: Inference response from the model.
        """
        print("prompt input:", prompt)
        
        # Mistral uses a different request format
        body = json.dumps(
            {
                "prompt": f"<s>[INST] {prompt} [/INST]",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            },
            ensure_ascii=False,
        )

        response = bedrock_runtime_client.invoke_model(
            body=body,
            modelId=inference_profile_arn,
        )
        
        # Parse the response based on Mistral's response format
        response_body = json.loads(response.get("body").read())
        message = response_body.get("outputs", [{}])[0].get("text", "")
        print("output message:", message)

        return (message,)


class BedrockMistral7B:
    @classmethod
    def INPUT_TYPES(s):
        # Try to get available Mistral 7B inference profile ARNs dynamically
        mistral_models = [
            "mistral.mistral-7b-instruct-v0:2"
        ]
        
        # Use safe version that doesn't raise errors
        available_arns = get_available_inference_profile_arns_safe(mistral_models)
        
        if available_arns:
            # Use first available ARN as default for text input
            default_arn = available_arns[0]
        else:
            # Manual text input if no profiles available
            print("Warning: No Mistral 7B inference profiles found. Using manual ARN input.")
            default_arn = "Enter Mistral 7B inference profile ARN"
        
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
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "round": 0.001,
                        "display": "slider",
                    },
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 50,
                        "min": 0,
                        "max": 200,
                        "step": 1,
                        "round": 1,
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
        Invokes the Mistral 7B model to run an inference using the input
        provided in the request body.

        :param prompt: The prompt that you want Mistral to complete.
        :param inference_profile_arn: The ARN of the inference profile to use.
        :return: Inference response from the model.
        """
        print("prompt input:", prompt)
        
        # Mistral uses a different request format
        body = json.dumps(
            {
                "prompt": f"<s>[INST] {prompt} [/INST]",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            },
            ensure_ascii=False,
        )

        response = bedrock_runtime_client.invoke_model(
            body=body,
            modelId=inference_profile_arn,
        )
        
        # Parse the response based on Mistral's response format
        response_body = json.loads(response.get("body").read())
        message = response_body.get("outputs", [{}])[0].get("text", "")
        print("output message:", message)

        return (message,)

# Required for ComfyUI to register the nodes
NODE_CLASS_MAPPINGS = {
    "Amazon Bedrock - Claude 4": BedrockClaude4,
    "Amazon Bedrock - Claude 3.5 Sonnet": BedrockClaude35Sonnet,
    "Amazon Bedrock - Claude 3 Haiku": BedrockClaude3Haiku,
    "Amazon Bedrock - Llama 4": BedrockLlama4,
    "Amazon Bedrock - Mistral Large": BedrockMistralLarge,
    "Amazon Bedrock - Mistral 7B": BedrockMistral7B,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Amazon Bedrock - Claude 4": "Amazon Bedrock - Claude 4",
    "Amazon Bedrock - Claude 3.5 Sonnet": "Amazon Bedrock - Claude 3.5 Sonnet",
    "Amazon Bedrock - Claude 3 Haiku": "Amazon Bedrock - Claude 3 Haiku",
    "Amazon Bedrock - Llama 4": "Amazon Bedrock - Llama 4",
    "Amazon Bedrock - Mistral Large": "Amazon Bedrock - Mistral Large",
    "Amazon Bedrock - Mistral 7B": "Amazon Bedrock - Mistral 7B",
}

print("Fixed bedrock.py loaded with safe error handling for ComfyUI environment")
print("If you see 'Warning: No inference profiles found', configure AWS credentials in ComfyUI")
