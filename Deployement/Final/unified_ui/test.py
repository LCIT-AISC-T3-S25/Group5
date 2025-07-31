import requests

headers = {"Authorization": "Bearer DCGAN_vm_access_token_123"}
res = requests.post("http://192.168.2.87:8000/generate", headers=headers)
print(res.status_code, res.json())
