from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from gradio_client import Client, handle_file
import shutil
import uuid
import os
import torch
import numpy as np
import cv2
from PIL import Image, ImageFilter
import mediapipe as mp

app = FastAPI(title="API VTON Avanzada Gratis")

OS_TEMP_DIR = "./temp"
os.makedirs(OS_TEMP_DIR, exist_ok=True)

HF_SPACE_URL = "fashn-ai/fashn-vton-1.5"
HF_TOKEN = None  # Agrega tu token si lo deseas: "hf_xxxxxxxxxxxxxxxxxxxx"


# ==========================================
# 1. MÁSCARA DEL ROSTRO (MEDIAPIPE)
# ==========================================
def obtener_mascara_rostro(imagen_pil):
    """Detecta el rostro/cabello para no distorsionarlo."""
    img_np = np.array(imagen_pil)
    h, w, _ = img_np.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    mp_face = mp.solutions.face_mesh
    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                points = [[int(lm.x * w), int(lm.y * h)] for lm in face_landmarks.landmark]
                hull = cv2.convexHull(np.array(points))
                cv2.fillConvexPoly(mask, hull, 255)
                
    if np.sum(mask) == 0:
        cv2.rectangle(mask, (int(w*0.3), int(h*0.05)), (int(w*0.7), int(h*0.35)), 255, -1)

    kernel = np.ones((15, 15), np.uint8)
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)
    return Image.fromarray(mask_dilated).convert("L").filter(ImageFilter.GaussianBlur(radius=10))


# ==========================================
# 2. ENDPOINTS DE FASTAPI
# ==========================================
@app.get("/")
def home():
    return {"status": "Servidor VTON Completo Activo"}

@app.post("/api/v1/try-on-completo")
async def try_on_completo(
    foto_persona: UploadFile = File(...),
    prenda_top: UploadFile = File(...),
    prenda_bottom: UploadFile = File(...)
):
    try:
        session_id = str(uuid.uuid4())
        persona_path = f"{OS_TEMP_DIR}/{session_id}_persona.jpg"
        top_path = f"{OS_TEMP_DIR}/{session_id}_top.jpg"
        bottom_path = f"{OS_TEMP_DIR}/{session_id}_bottom.jpg"
        final_output_path = f"{OS_TEMP_DIR}/{session_id}_final.png"

        # Guardar imágenes recibidas de la app
        with open(persona_path, "wb") as buffer:
            shutil.copyfileobj(foto_persona.file, buffer)
        with open(top_path, "wb") as buffer:
            shutil.copyfileobj(prenda_top.file, buffer)
        with open(bottom_path, "wb") as buffer:
            shutil.copyfileobj(prenda_bottom.file, buffer)

        persona_orig = Image.open(persona_path).convert("RGB")

        # A. Crear la máscara del rostro para protección al final
        mascara_rostro = obtener_mascara_rostro(persona_orig)

        # B. Conectar con Hugging Face
        client = Client(HF_SPACE_URL, hf_token=HF_TOKEN)

        # C. PASO 1: Procesar Prenda Superior (Top)
        res_step1_path = client.predict(
            model_input={"background": handle_file(persona_path), "layers": [], "composite": None},
            garm_img=handle_file(top_path),
            garment_des="top",
            is_checked=True,
            is_checked_crop=False,
            denoise_steps=30,
            seed=42,
            api_name="/process"
        )

        # D. PASO 2: Procesar Prenda Inferior (Bottom) sobre la imagen del Paso 1
        res_step2_path = client.predict(
            model_input={"background": handle_file(res_step1_path), "layers": [], "composite": None},
            garm_img=handle_file(bottom_path),
            garment_des="bottom",
            is_checked=True,
            is_checked_crop=False,
            denoise_steps=30,
            seed=42,
            api_name="/process"
        )

        # E. PASO 3: Restauración Facial
        res_step2_pil = Image.open(res_step2_path).convert("RGB")
        resultado_final = Image.composite(persona_orig, res_step2_pil, mascara_rostro)
        resultado_final.save(final_output_path)

        # Retornar la imagen final a la App Móvil
        return FileResponse(final_output_path, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en procesamiento VTON: {str(e)}")
