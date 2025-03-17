import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
import runpod
from io import BytesIO
import base64
from huggingface_hub import login


def handler(event):
    input = event['input']
    image = input.get('image')
    HF_API_KEY = input.get('hf_api')
    assistant_start = "<|start_header_id|>assistant<|end_header_id>"

    if HF_API_KEY:
        login(token=HF_API_KEY)
    else:
        raise ValueError("Hugging Face API key is missing. Set HF_API_KEY as an environment variable.")
    #Make string to pillow image
    # b64decode will make it in bytes b'jdhkfghfh' format
    # BytesIO will
    image = Image.open(BytesIO(base64.b64decode(image)))

    # Placeholder for a task; replace with image or text generation logic as needed
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "Analyze the given image and provide the following details: 1. Count of people present in the image. 2. Identify activities happening in the image (e.g., criminal avtivity, detect weapons, working on a computer, having a conversation, idle, etc.).\n 3. Determine whether individuals are is alone, in groups, or not engaged in work, or are walking. Identify there pose\n4. **Important**: Provide detailed observations about the environment, and identify interaction between various people and objects.\n 5. Highlight any unusual or suspicious behavior (if any).\n\nFormat the response as a structured report for easy understanding."}
        ]}
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)
    output = model.generate(**inputs, max_new_tokens=1024)
    responce = clean_responce(processor.decode(output[0]))
    return responce


def clean_responce(text):
    assistant_start = "<|start_header_id|>assistant<|end_header_id|>"  # Marks the start of the response
    assistant_end = "<|eot_id|>"
    request_end = "<|eot_id|>"
    if assistant_start in text:
        response_part = text.split(assistant_start, 1)[1]
        if assistant_end in response_part:
            clean_response = response_part.split(assistant_end, 1)[0].strip()
        else:
            clean_response = response_part.strip()
    elif request_end in text:
        # Fallback: if assistant_start is missing, try splitting after request_end
        clean_response = text.split(request_end, 1)[1].strip()
    else:
        # Last resort: return trimmed output if no markers are found
        clean_response = text.strip()
    
    return clean_response


if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})

