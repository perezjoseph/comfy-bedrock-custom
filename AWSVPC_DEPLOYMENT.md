# AWS VPC Mode Deployment Guide

This guide covers deploying ComfyUI with Bedrock nodes in **awsvpc networking mode** environments like ECS Tasks, EKS Pods, and Fargate.

## 🔍 **What is awsvpc Mode?**

In awsvpc mode, containers get their own network interface and cannot access the EC2 instance metadata service (IMDS) at `169.254.169.254`. This affects how AWS credentials are resolved.

## 🚀 **Deployment Scenarios**

### **1. ECS Tasks with awsvpc Mode**

#### Task Definition Configuration:
```json
{
  "family": "comfyui-bedrock",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "taskRoleArn": "arn:aws:iam::ACCOUNT:role/ComfyUIBedrockTaskRole",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "comfyui",
      "image": "your-comfyui-image",
      "environment": [
        {
          "name": "AWS_DEFAULT_REGION",
          "value": "us-east-1"
        }
      ],
      "portMappings": [
        {
          "containerPort": 8188,
          "protocol": "tcp"
        }
      ]
    }
  ]
}
```

#### Required IAM Task Role Policy:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:ListInferenceProfiles",
        "bedrock:GetInferenceProfile"
      ],
      "Resource": "*"
    }
  ]
}
```

### **2. EKS Pods with IRSA**

#### Service Account with IAM Role:
```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: comfyui-bedrock-sa
  namespace: default
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::ACCOUNT:role/ComfyUIBedrockRole
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: comfyui-bedrock
spec:
  replicas: 1
  selector:
    matchLabels:
      app: comfyui-bedrock
  template:
    metadata:
      labels:
        app: comfyui-bedrock
    spec:
      serviceAccountName: comfyui-bedrock-sa
      containers:
      - name: comfyui
        image: your-comfyui-image
        env:
        - name: AWS_DEFAULT_REGION
          value: "us-east-1"
        ports:
        - containerPort: 8188
```

### **3. AWS Fargate**

Fargate automatically uses awsvpc mode. Follow the ECS configuration above, ensuring:
- `requiresCompatibilities: ["FARGATE"]`
- Proper Task Role with Bedrock permissions
- `AWS_DEFAULT_REGION` environment variable set

## 🔧 **Environment Variables**

Set these environment variables in your container:

```bash
# Required: AWS Region
AWS_DEFAULT_REGION=us-east-1

# Optional: Alternative region variable
AWS_REGION=us-east-1

# Optional: For debugging
AWS_SDK_LOAD_CONFIG=1
```

## 🛠️ **Troubleshooting**

### **Common Issues:**

1. **"No credentials found" error:**
   - Ensure Task Role/Service Account is properly attached
   - Verify IAM permissions include `bedrock:*` actions
   - Check that `AWS_DEFAULT_REGION` is set

2. **"Region not found" error:**
   - Set `AWS_DEFAULT_REGION` environment variable
   - Ensure the region supports Bedrock service

3. **"Access denied" error:**
   - Verify IAM role has Bedrock permissions
   - Check if Bedrock is enabled in your AWS account
   - Ensure you're using the correct region

### **Testing Credentials:**

Run this in your container to test AWS credentials:
```python
import boto3
try:
    sts = boto3.client('sts')
    identity = sts.get_caller_identity()
    print(f"Account: {identity['Account']}")
    print(f"ARN: {identity['Arn']}")
    print("✅ Credentials working!")
except Exception as e:
    print(f"❌ Credential error: {e}")
```

## 📋 **Required IAM Permissions**

Minimum permissions for the Task Role/Service Account:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel"
      ],
      "Resource": [
        "arn:aws:bedrock:*::foundation-model/*",
        "arn:aws:bedrock:*:*:inference-profile/*"
      ]
    },
    {
      "Effect": "Allow", 
      "Action": [
        "bedrock:ListInferenceProfiles",
        "bedrock:GetInferenceProfile"
      ],
      "Resource": "*"
    }
  ]
}
```

## 🔍 **Verification Steps**

1. **Check container logs** for credential-related messages
2. **Verify environment variables** are set correctly
3. **Test IAM permissions** using AWS CLI in the container
4. **Confirm region availability** for Bedrock service

## 💡 **Best Practices**

- Use **least privilege** IAM policies
- Set **explicit regions** via environment variables
- Monitor **CloudTrail logs** for API calls
- Use **AWS Secrets Manager** for sensitive configuration
- Implement **proper error handling** in your workflows

## 🆘 **Support**

If you encounter issues:
1. Check the container logs for detailed error messages
2. Verify your IAM role/policy configuration
3. Test credentials using the verification steps above
4. Ensure Bedrock is available in your chosen region
