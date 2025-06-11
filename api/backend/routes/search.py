from http.client import HTTPException
from fastapi import UploadFile, Form, File
import time
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from backend.services.clip_search import load_model, load_gallery_features, load_model_by_name
from backend.services.clip_search import run_inference
from backend.config import MODEL_CONFIGS
router = APIRouter()  

@router.post("/search")
async def search_endpoint(
    image: UploadFile = File(...),
    clip_arch: str = Form(...),
    model: str = Form(...),
    use_finetuned: bool = Form(False),
    top_k: int = Form(10)
):
    start_time = time.time()

    # Log model selection
    print("Received API request")
    print(f"Model selected: {model}, Use fine-tuned: {use_finetuned}")
    print(f"Filename: {image.filename}")

    # Log image content details
    content = await image.read()
    print(f"  • Image Size: {len(content)} bytes")

    # Rewind image for downstream processing
    image.file.seek(0)

    # Inference starts here
    model_result = load_model_by_name(clip_arch, model)
    #print(f"model_result : {model_result}")

    if model_result is None:
       raise HTTPException(400, f"Model '{model}' could not be loaded. Please verify model config or checkpoint path.")

    model_obj, preprocess, gallery_feats, gallery_paths = model_result
    top_k = run_inference(model_obj, preprocess, image.file, gallery_feats, gallery_paths,top_k=top_k)

    #  Log results
    print(f"  • Top result: {top_k[0] if top_k else 'None'}")
    print(f"  • Total Results: {len(top_k)}")
    print(f" Took {time.time() - start_time:.2f}s\n")

    return {"similar_images": top_k}

