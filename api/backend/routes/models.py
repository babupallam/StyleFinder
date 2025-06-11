from fastapi import APIRouter, HTTPException, Path
from backend.config import MODEL_CONFIGS  

router = APIRouter()


@router.get("/models")
async def get_model_configs():
    return MODEL_CONFIGS


@router.get("/model-description/{arch:path}/{model_name}")
def get_model_description(arch: str, model_name: str):
    print(f"→ Received arch: {arch}, model: {model_name}")
    
    arch = arch.strip()
    model_name = model_name.strip()

    if arch not in MODEL_CONFIGS:
        print(" Architecture not found:", arch)
        raise HTTPException(status_code=404, detail="Architecture not found")

    if model_name not in MODEL_CONFIGS[arch]:
        print(" Model not found:", model_name)
        raise HTTPException(status_code=404, detail="Model not found")

    description = MODEL_CONFIGS[arch][model_name].get("description", {})
    return {"description": description}
