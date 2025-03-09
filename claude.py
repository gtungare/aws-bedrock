import boto3
import json

"""
This script generates a poem in the style of Shakespeare on the topic of Generative AI 
using AWS Bedrock's AI21 model (J2 Mid Version 1). The poem is created based on a textual 
prompt, and the response is printed to the console.

The following steps are executed:
1. The script sends a prompt requesting the model to write a poem about Generative AI 
   in the style of Shakespeare.
2. The model generates a response, and the poem text is extracted from the response.
3. The generated poem is printed to the console.

Dependencies:
- boto3: AWS SDK for Python to interact with the AWS Bedrock service.
- json: To format the payload and parse the response.

Configuration:
- The prompt specifies the content (a poem on Generative AI) and style (Shakespeare).
- The model parameters (maxTokens, temperature, topP) control the text generation quality 
  and creativity.

Output:
- The generated poem is printed to the console.

Usage:
Run 

prompt_data="""
Act as a Shakespeare and write a poem on Genertaive AI
"""

bedrock=boto3.client(service_name="bedrock-runtime")

payload={
    "prompt":prompt_data,
    "maxTokens":512,
    "temperature":0.8,
    "topP":0.8
}
body = json.dumps(payload)
model_id = "ai21.j2-mid-v1"
response = bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json",
)

response_body = json.loads(response.get("body").read())
response_text = response_body.get("completions")[0].get("data").get("text")
print(response_text)