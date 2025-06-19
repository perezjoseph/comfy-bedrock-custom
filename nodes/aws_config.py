"""
AWS Configuration helper for awsvpc mode environments
Handles credential resolution for ECS Tasks, EKS Pods, and other containerized environments
"""

import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError


def get_aws_config():
    """
    Get AWS configuration optimized for awsvpc mode environments.
    Returns a dictionary with region and any additional config needed.
    """
    config = {}
    
    # Try to get region from various sources
    region = (
        os.environ.get('AWS_DEFAULT_REGION') or 
        os.environ.get('AWS_REGION') or
        os.environ.get('AWS_DEFAULT_REGION') or
        'us-east-1'  # Default fallback
    )
    
    config['region_name'] = region
    
    # Check if we're in a containerized environment
    if os.path.exists('/.dockerenv') or os.environ.get('AWS_EXECUTION_ENV'):
        config['containerized'] = True
        print(f"Detected containerized environment, using region: {region}")
    else:
        config['containerized'] = False
    
    return config


def test_aws_credentials():
    """
    Test if AWS credentials are properly configured.
    Returns True if credentials work, False otherwise.
    """
    try:
        # Try to create a simple STS client to test credentials
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"AWS credentials verified. Account: {identity.get('Account')}, ARN: {identity.get('Arn')}")
        return True
    except NoCredentialsError:
        print("No AWS credentials found. Please configure credentials.")
        return False
    except ClientError as e:
        print(f"AWS credential error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error testing credentials: {e}")
        return False


def get_recommended_setup():
    """
    Provide setup recommendations based on the environment.
    """
    config = get_aws_config()
    
    recommendations = []
    
    if config.get('containerized'):
        recommendations.extend([
            "🐳 Containerized Environment Detected",
            "",
            "For ECS Tasks (awsvpc mode):",
            "  • Attach an IAM Task Role to your ECS Task Definition",
            "  • Ensure the role has bedrock:InvokeModel permissions",
            "  • Set AWS_DEFAULT_REGION environment variable",
            "",
            "For EKS Pods:",
            "  • Use IAM Roles for Service Accounts (IRSA)",
            "  • Annotate your service account with the IAM role ARN",
            "  • Set AWS_DEFAULT_REGION environment variable",
            "",
            "For Fargate:",
            "  • Attach an IAM Task Role with Bedrock permissions",
            "  • Ensure awsvpc network mode is configured",
        ])
    else:
        recommendations.extend([
            "🖥️  Standard Environment Detected",
            "",
            "Recommended credential setup:",
            "  • Use AWS CLI: aws configure",
            "  • Or set environment variables:",
            "    - AWS_ACCESS_KEY_ID",
            "    - AWS_SECRET_ACCESS_KEY", 
            "    - AWS_DEFAULT_REGION",
            "  • Or use IAM Instance Profile (if on EC2)",
        ])
    
    return "\n".join(recommendations)


if __name__ == "__main__":
    # Test the configuration
    print("=== AWS Configuration Test ===")
    config = get_aws_config()
    print(f"Configuration: {config}")
    print()
    
    print("=== Credential Test ===")
    if test_aws_credentials():
        print("✅ AWS credentials are working!")
    else:
        print("❌ AWS credentials need to be configured")
        print()
        print(get_recommended_setup())
