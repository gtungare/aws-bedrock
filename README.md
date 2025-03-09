# Colgate Toothpaste Poem Generator using AWS Bedrock

This Python script utilizes AWS Bedrock to generate a poem about Colgate Toothpaste. The script interacts with the `meta.llama2-70b-chat-v1` model hosted on AWS, leveraging the Bedrock service to generate text based on a prompt.

## Prerequisites

Before running the script, ensure you have the following installed and configured:

- Python 3.x
- [AWS SDK for Python (Boto3)](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- AWS credentials configured via the AWS CLI or environment variables
- AWS Bedrock access with the required model (`meta.llama2-70b-chat-v1`)

## Installation

1. Install the required Python packages:

    ```bash
    pip install boto3
    ```

2. Ensure that your AWS credentials are configured. You can configure AWS CLI with the following command:

    ```bash
    aws configure
    ```

## Script Overview

The script performs the following operations:

1. Creates a `boto3` client to interact with the AWS Bedrock service.
2. Defines a prompt asking the model to generate a poem about Colgate Toothpaste.
3. Sends a request to the AWS Bedrock service with a JSON payload containing the prompt and other parameters (`max_gen_len`, `temperature`, `top_p`).
4. Receives and parses the response from the model.
5. Prints the generated poem to the console.

## Code Breakdown

```python
import boto3
import json

prompt_data = """
Act as a domain expert and write a poem on Colgate Toothpaste
"""

# Initialize Bedrock client
bedrock = boto3.client(service_name="bedrock-runtime")

# Define the payload with prompt data and configuration
payload = {
    "prompt": "[INST]" + prompt_data + "[/INST]",
    "max_gen_len": 512,
    "temperature": 0.5,
    "top_p": 0.9
}

body = json.dumps(payload)
model_id = "meta.llama2-70b-chat-v1"

# Invoke model to generate text
response = bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json"
)

# Parse the response and extract generated text
response_body = json.loads(response.get("body").read())
response_text = response_body['generation']

# Print the generated poem
print(response_text)
