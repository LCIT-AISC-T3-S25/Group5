import requests
import yaml
from PIL import Image
from io import BytesIO

# Load config.yaml
with open("utils/config.yaml", "r") as f:
    config = yaml.safe_load(f)

VM_IP = config["vm_ip"]
PORTS = config["ports"]
TOKENS = config["tokens"]

def call_transformer_model(text):
    try:
        res = requests.post(f"{VM_IP}:{PORTS['transformer']}/predict", json={
            "auth_token": TOKENS["transformer"],
            "text": text
        })
        print("Transformer Response:", res.status_code, res.text)
        if res.status_code == 200:
            return res.json()
        return None
    except Exception as e:
        print("Transformer error:", e)
        return None

def call_dcgan_model():
    try:
        headers = {"Authorization": f"Bearer {TOKENS['dcgan']}"}
        res = requests.post(f"{VM_IP}:{PORTS['dcgan']}/generate", headers=headers)
        print("DCGAN Response:", res.status_code, res.text)
        if res.status_code == 200:
            img_path = res.json().get("image_path")
            return f"{VM_IP}:{PORTS['dcgan']}/{img_path}"
        return None
    except Exception as e:
        print("DCGAN error:", e)
        return None

def call_ddim_model(prompt):
    try:
        res = requests.post(f"{VM_IP}:{PORTS['ddim']}/generate", json={"prompt": prompt})
        print("DDIM Response:", res.status_code, res.text)
        if res.status_code == 200:
            return f"{VM_IP}:{PORTS['ddim']}/{res.json().get('file')}"
        return None
    except Exception as e:
        print("DDIM error:", e)
        return None

def call_rag_model(query):
    try:
        res = requests.post(f"{VM_IP}:{PORTS['rag']}/chat", json={"query": query})
        print("RAG Response:", res.status_code, res.text)
        if res.status_code == 200:
            return res.json().get("response")
        return None
    except Exception as e:
        print("RAG error:", e)
        return None
