# Stable Diffusion em português
Exemplo de como rodar localmente uma inferência no modelo de AI Stable Diffusion desenvolvido por [stability.ai](https://stability.ai/blog/stable-diffusion-public-release).

# Instalação
Usando conda:
```
conda env create --name=stable-difusion --file environment.yaml
```

# Execução
Ative o environment:
```
conda activate stable-difusion
```

Crie uma conta em [Hugging Face](https://huggingface.co/) e crie um token de acesso [aqui](https://huggingface.co/settings/tokens).

Execute com o prompt desejado
```
python inference_pt_br.py --output-folder <OUTPUT FOLDER> --prompt <PROMPT> --access-token <HUGGING FACE TOKEN>
```