import boto3
import json
import base64
import os

"""
This script generates a 4K HD image of a beach with a blue sky and rainy season using AWS Bedrock's 
Stable Diffusion XL model. The image is generated based on a textual prompt and is then saved to 
a specified output directory.

The following steps are executed:
1. The script sends a prompt to the AWS Bedrock service for generating the image.
2. The response from the model contains the image in base64 encoding.
3. The script decodes the base64-encoded image and saves it as a PNG file in the 'output' directory.

Dependencies:
- boto3: AWS SDK for Python to interact with the AWS Bedrock service.
- json: To format the payload and parse the response.
- base64: For decoding the base64-encoded image response.
- os: To create the output directory and handle file I/O.

Configuration:
- The prompt specifies the content and style of the image (a beach with a blue sky and rainy season).
- The image size is set to 512x512 pixels, and the model configuration (cfg_scale, seed, steps) 
  controls the image generation quality and variability.

Output:
- A PNG image is saved to the 'output' directory with the filename 'generated-img.png'.

Usage:
Run the script to generate the image. The output will be saved in the 'output' directory as a PNG file.
"""

prompt_data = """
provide me an 4k hd image of a beach, also use a blue sky rainy season and
cinematic display
"""
prompt_template=[{"text":prompt_data,"weight":1}]
bedrock = boto3.client(service_name="bedrock-runtime")
payload = {
    "text_prompts":prompt_template,
    "cfg_scale": 10,
    "seed": 0,
    "steps":50,
    "width":512,
    "height":512

}

body = json.dumps(payload)
model_id = "stability.stable-diffusion-xl-v0"
response = bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json",
)

response_body = json.loads(response.get("body").read())
print(response_body)
artifact = response_body.get("artifacts")[0]
image_encoded = artifact.get("base64").encode("utf-8")
image_bytes = base64.b64decode(image_encoded)

# Save image to a file in the output directory.
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
file_name = f"{output_dir}/generated-img.png"
with open(file_name, "wb") as f:
    f.write(image_bytes)