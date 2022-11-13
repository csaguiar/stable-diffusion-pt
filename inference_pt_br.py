import click
import os
from diffusers import StableDiffusionPipeline
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DIFFUSION_MODEL_ID = "runwayml/stable-diffusion-v1-5"
"""
Translation to portuguese inspired by https://twitter.com/joao_gante
https://twitter.com/joao_gante/status/1560278844153548803
"""
TRANSLATION_MODEL_ID = "Narrativa/mbart-large-50-finetuned-opus-pt-en-translation"  # noqa


def load_translation_models(translation_model_id, access_token):
    tokenizer = AutoTokenizer.from_pretrained(
        translation_model_id,
        use_auth_token=access_token
    )
    text_model = AutoModelForSeq2SeqLM.from_pretrained(
        translation_model_id,
        use_auth_token=access_token
    )

    return tokenizer, text_model


def pipeline_generate(diffusion_model_id, access_token, prompt):
    pipe = StableDiffusionPipeline.from_pretrained(
        diffusion_model_id,
        use_auth_token=access_token
    )
    pipe = pipe.to("mps")

    # Recommended if your computer has < 64 GB of RAM
    pipe.enable_attention_slicing()

    return pipe


def translate(prompt, tokenizer, text_model):
    pt_tokens = tokenizer([prompt], return_tensors="pt")
    en_tokens = text_model.generate(
        **pt_tokens, max_new_tokens=100,
        num_beams=8, early_stopping=True
    )
    en_prompt = tokenizer.batch_decode(en_tokens, skip_special_tokens=True)
    print(f"translation: [PT] {prompt} -> [EN] {en_prompt[0]}")

    return en_prompt[0]


def generate_image(pipe, prompt):
    # First-time "warmup" pass (see explanation above)
    _ = pipe(prompt, num_inference_steps=1)

    return pipe(prompt).images[0]


@click.command()
@click.option('--output-folder', help='Where to save it')
@click.option('--prompt', required=True, help='Image description to generate')
@click.option(
    '--access-token', help='Hugging Face token',
    default=os.getenv("HUGGING_FACE_TOKEN")
)
@click.option(
    '--translation-model', help='Translation model ID',
    default=TRANSLATION_MODEL_ID
)
@click.option(
    '--diffusion-model', help='Diffusion model ID',
    default=DIFFUSION_MODEL_ID
)
def main(**kwargs):
    output_path = Path(kwargs["output_folder"])
    output_path.mkdir(exist_ok=True)
    prompt = kwargs["prompt"]
    translation_model_id = kwargs["translation_model"]
    diffusion_model_id = kwargs["diffusion_model"]
    access_token = kwargs["access_token"]

    tokenizer, text_model = load_translation_models(
        translation_model_id, access_token
    )
    prompt_en = translate(prompt, tokenizer, text_model)
    pipe = pipeline_generate(diffusion_model_id, access_token, prompt_en)
    image = generate_image(pipe, prompt)
    image.save(output_path / "inference.png")


if __name__ == "__main__":
    main()
