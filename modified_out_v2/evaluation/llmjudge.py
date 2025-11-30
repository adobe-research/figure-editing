# llmjudge.py
"""
This module provides the LLMJudge class, which uses the Azure GPT-4o service 
to evaluate AI-generated charts. It provides a quantitative score and reasoning 
by comparing the generated chart, a ground truth chart, and the editing instruction.
"""

import base64
import json
import os
import logging
from openai import AzureOpenAI
from typing import Optional, Tuple

# Updated prompt to include all three sub-scores and total score
_PROMPT_TEMPLATE = """
You are an expert AI assistant specializing in data visualization evaluation.
Your task is to evaluate how well an AI-generated chart follows a given text instruction. You will be given an instruction, a reference "Ground Truth" image, and the "Generated Image" to evaluate.

Evaluate the generated image based on the following three criteria:

1. **Instruction Following (score_instruction)**: How accurately was the specific instruction executed? (e.g., if asked to change color to orange, is it orange?)
2. **Content Preservation (score_preservation)**: Were all other elements of the chart preserved correctly without unwanted changes? (e.g., data values, labels, and titles are unchanged unless specified).
3. **Image Quality (score_quality)**: Is the generated image free of major artifacts, distortions, or unreadable text?

For each of the above, assign a score from 1 (very poor) to 5 (excellent).  
Then compute a **total score (score)** as the average of the three above, rounded to the nearest integer.

Your response MUST be a JSON object with the following keys:
- "score_instruction": Integer [1–5]
- "score_preservation": Integer [1–5]
- "score_quality": Integer [1–5]
- "score": Integer [1–5], the average of the above
- "reasoning": One-sentence explanation justifying the scores

Example Response:
{
    "score_instruction": 5,
    "score_preservation": 4,
    "score_quality": 5,
    "score": 5,
    "reasoning": "The instruction was followed perfectly, content was mostly preserved, and the image quality is excellent."
}
"""

class LLMJudge:
    """
    A judge that uses GPT-4o to evaluate the quality of chart edits.
    """
    def __init__(self, api_key: str, azure_endpoint: str, api_version: str = "2024-05-01-preview", model_deployment_name: str = "gpt-4o"):
        """
        Initializes the Azure OpenAI client.

        Args:
            api_key (str): The Azure OpenAI API key.
            azure_endpoint (str): The endpoint for the Azure OpenAI service.
            api_version (str): The API version to use.
            model_deployment_name (str): Your deployment name for GPT-4o on Azure.
        """
        if not api_key:
            raise ValueError("API key cannot be empty.")
        if not azure_endpoint:
            raise ValueError("Azure endpoint cannot be empty.")

        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint
        )
        self.model = model_deployment_name
        print(f"LLM Judge initialized with model deployment '{self.model}' at endpoint '{azure_endpoint}'.")

    def _encode_image_to_base64(self, image_path: str) -> Optional[str]:
        """Encodes a local image file into a base64 string."""
        try:
            with open(image_path, "rb") as image_file:
                encoded_bytes = base64.b64encode(image_file.read())
                return encoded_bytes.decode('utf-8')
        except (FileNotFoundError, IOError) as e:
            logging.error(f"[Error] Failed to open or encode image '{image_path}': {e}")
            return None

    def evaluate(self, gt_path: str, test_path: str, instruction: str) -> Tuple[float, float, float, float, str]:
        """
        Evaluates the quality of a generated image using GPT-4o.

        Args:
            gt_path (str): Path to the ground truth image.
            test_path (str): Path to the model-generated image to be tested.
            instruction (str): The editing instruction used to generate the image.

        Returns:
            tuple: (
                instruction_score: float,
                preservation_score: float,
                quality_score: float,
                total_score: float,
                reasoning: str
            )
            Returns (nan, nan, nan, nan, error_message) on failure.
        """
        gt_base64 = self._encode_image_to_base64(gt_path)
        test_base64 = self._encode_image_to_base64(test_path)

        if not gt_base64 or not test_base64:
            return float("nan"), float("nan"), float("nan"), float("nan"), "Image encoding failed"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _PROMPT_TEMPLATE},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"**Instruction:**\n{instruction}"},
                            {"type": "text", "text": "\n**Reference Image (Ground Truth):**"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{gt_base64}"}},
                            {"type": "text", "text": "\n**Generated Image (to be evaluated):**"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{test_base64}"}},
                        ],
                    }
                ],
                max_tokens=200,
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            result_json = json.loads(response.choices[0].message.content)
            instruction_score = float(result_json.get("score_instruction", float("nan")))
            preservation_score = float(result_json.get("score_preservation", float("nan")))
            quality_score = float(result_json.get("score_quality", float("nan")))
            total_score = float(result_json.get("score", float("nan")))
            reasoning = result_json.get("reasoning", "No reasoning provided.")

            return instruction_score, preservation_score, quality_score, total_score, reasoning

        except Exception as e:
            print(f"[Error] API call to GPT-4o failed: {e}")
            return float("nan"), float("nan"), float("nan"), float("nan"), str(e)