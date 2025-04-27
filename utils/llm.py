import os
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from utils.scripts import get_prompt
from llama_cpp import Llama
import os
from huggingface_hub import hf_hub_download

selected_llm = 'Mistral-7B-OpenOrca' 
model_dic = {"Mistral-7B":{"HF_REPO_NAME":"TheBloke/Mistral-7B-Instruct-v0.1-GGUF","HF_MODEL_NAME":"mistral-7b-instruct-v0.1.Q4_K_M.gguf"},
           "Mistral-7B-OpenOrca":{"HF_REPO_NAME":"TheBloke/Mistral-7B-OpenOrca-GGUF","HF_MODEL_NAME":"mistral-7b-openorca.Q5_K_M.gguf"},
             "Llama-2-13B-Chat":{"HF_REPO_NAME":"TheBloke/Llama-2-13B-chat-GGUF","HF_MODEL_NAME":"llama-2-13b-chat.Q4_K_S.gguf"}
             }

HF_REPO_NAME = model_dic[selected_llm]['HF_REPO_NAME']
HF_MODEL_NAME = model_dic[selected_llm]['HF_MODEL_NAME']
LOCAL_DIR_NAME = "models"

os.makedirs(LOCAL_DIR_NAME, exist_ok=True)
model_path = hf_hub_download(
    repo_id=HF_REPO_NAME, filename=HF_MODEL_NAME, local_dir=LOCAL_DIR_NAME
)

def get_response_from_llm(dat):
    llm = Llama(
        model_path=model_path,
        n_threads=2,
        n_batch=512, 
        n_gpu_layers=30, 
        n_ctx=4096, 
    )
    prompt = get_prompt(0)
    response = llm(prompt, stream=True, stop=["\n\n"], temperature=0, max_tokens=200)
    generated_text = ""

    for output in response:
        result = output['choices'][0]['text']

        if result.strip():
            generated_text += result

    return generated_text


#A ferrugem é uma doença do feijão causada por um fungo, Puccinia psittacina. Esta doença pode ser causada por condições ambientais desfavoráveis, como temperaturas muito baixas ou altas, ou por falta de nutrientes no solo. Além disso, a falta de aplicação de fungicidas pode também ser um fator que contribui para a ocorrência da ferrugem.