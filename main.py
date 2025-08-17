from pathlib import Path

import modal
import requests
import torch
from loguru import logger
from PIL import Image, ImageFile
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_VOL = modal.Volume.from_name(
    "analysis-vision-models-model-cache", create_if_missing=True
)
MODEL_VOL_PATH = Path("/models")


image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install(["libgl1-mesa-glx", "wget", "libcudnn8", "libcudnn8-dev"])
    .uv_sync()
)

app = modal.App("analysis-vision-models", image=image)


@app.cls(
    gpu="L40S",
    timeout=900,
    retries=0,
    scaledown_window=20,
    volumes={MODEL_VOL_PATH: MODEL_VOL},
)
class AnalysisVisionModel:
    @modal.enter()
    def load_model(self) -> None:
        self._MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
        self._DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        self._processor = AutoProcessor.from_pretrained(self._MODEL_ID)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self._MODEL_ID, torch_dtype=torch.float16
        ).to(self._DEVICE)

        logger.success(f"Model loaded to device: {self._DEVICE}")

    def _analyze_image_with_smolvlm(
        self, *, image: ImageFile, question: str, max_tokens: int = 200
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": question}],
            }
        ]

        prompt = self._processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self._processor(text=prompt, images=[image], return_tensors="pt").to(
            self._DEVICE
        )

        generated_ids = self._model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.3
        )

        return self._processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )[0]

    @modal.method()
    def analyze_image(
        self, image_url: str, question: str, max_tokens: int = 200
    ) -> str:
        try:
            response = requests.get(image_url, stream=True)
            response.raise_for_status()
            image = Image.open(response.raw)

            result = self._analyze_image_with_smolvlm(
                image=image, question=question, max_tokens=max_tokens
            )
            return result

        except Exception as e:
            return f"Erro processing image: {str(e)}"


@app.local_entrypoint()
def test_local_analysis():
    model = AnalysisVisionModel()

    receipt_url = "https://raw.githubusercontent.com/mistralai/cookbook/main/mistral/ocr/receipt.png"

    document_questions = [
        "What type of document is this?",
        "What is the total amount?",
        "What items can you identify?",
    ]

    logger.info("Starting image analysis...")

    for question in document_questions:
        try:
            answer = model.analyze_image.remote(receipt_url, question, max_tokens=150)
            logger.info(f"Question: {question}")
            logger.info(f"Answer: {answer}")
            logger.info("-" * 40)
        except Exception as e:
            logger.error(f"Error on '{question}': {str(e)}")
