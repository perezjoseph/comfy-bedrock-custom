import requests
from retry import retry
import boto3
import os


MAX_RETRY = 3


@retry(tries=MAX_RETRY)
def get_client(service_name, clients={}):
    if service_name in clients:
        return clients[service_name]

    try:
        # First attempt: Use standard AWS credential chain
        # This works with Task Roles, environment variables, credential files, etc.
        clients[service_name] = boto3.client(service_name=service_name)
        print(f"Successfully created {service_name} client using standard credential chain")
        return clients[service_name]
        
    except Exception as e:
        print(f"Standard credential chain failed for {service_name}: {e}")
        
        # Second attempt: Try with explicit region from environment
        region = os.environ.get('AWS_DEFAULT_REGION') or os.environ.get('AWS_REGION')
        if region:
            try:
                clients[service_name] = boto3.client(service_name=service_name, region_name=region)
                print(f"Successfully created {service_name} client using region from environment: {region}")
                return clients[service_name]
            except Exception as e2:
                print(f"Failed to create client with environment region {region}: {e2}")
        
        # Third attempt: Try IMDS (only works on EC2, not in awsvpc mode)
        try:
            print("Attempting to get region from EC2 instance metadata...")
            # Use IMDSv2 token
            response = requests.put(
                "http://169.254.169.254/latest/api/token",
                headers={
                    "X-aws-ec2-metadata-token-ttl-seconds": "21600",
                },
                timeout=2  # Short timeout since this likely won't work in awsvpc mode
            )
            token = response.text
            response = requests.get(
                "http://169.254.169.254/latest/meta-data/placement/region",
                headers={
                    "X-aws-ec2-metadata-token": token,
                },
                timeout=2
            )
            detected_region = response.text
            boto3.setup_default_session(region_name=detected_region)
            clients[service_name] = boto3.client(service_name=service_name)
            print(f"Successfully created {service_name} client using IMDS detected region: {detected_region}")
            return clients[service_name]
            
        except Exception as e3:
            print(f"IMDS region detection failed (expected in awsvpc mode): {e3}")
        
        # Fourth attempt: Try common regions as fallback
        fallback_regions = ['us-east-1', 'us-west-2', 'eu-west-1']
        for fallback_region in fallback_regions:
            try:
                clients[service_name] = boto3.client(service_name=service_name, region_name=fallback_region)
                print(f"Successfully created {service_name} client using fallback region: {fallback_region}")
                return clients[service_name]
            except Exception as e4:
                print(f"Failed to create client with fallback region {fallback_region}: {e4}")
                continue
        
        # If all attempts fail, raise the original error
        raise Exception(f"Failed to create {service_name} client. Please ensure AWS credentials are configured properly. "
                       f"For awsvpc mode (ECS/EKS), ensure Task Role or Service Account has proper permissions. "
                       f"Original error: {e}")
    
    return clients[service_name]
