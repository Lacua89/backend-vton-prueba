import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from gradio_client import Client, handle_file
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "API VTON Activa"}

@app.post("/api/v1/try-on-completo")
async def try_on(
    foto_persona: UploadFile = File(...),
    prenda_top: UploadFile = File(...),
    prenda_bottom: UploadFile = File(...)
):
    try:
        hf_token = os.getenv("HF_TOKEN")
        
        persona_path = "temp_person.jpg"
        top_path = "temp_top.jpg"
        bottom_path = "temp_bottom.jpg"
        combined_path = "temp_outfit_combined.jpg"
        
        # 1. Guardar archivos recibidos desde Flutter
        with open(persona_path, "wb") as f:
            f.write(await foto_persona.read())
        with open(top_path, "wb") as f:
            f.write(await prenda_top.read())
        with open(bottom_path, "wb") as f:
            f.write(await prenda_bottom.read())

        # 2. Combinar prendas verticalmente con PIL (Top arriba, Bottom abajo)
        top_img = Image.open(top_path).convert("RGB")
        bottom_img = Image.open(bottom_path).convert("RGB")

        # Ajustar anchos para que ambas imágenes coincidan exactamente
        target_width = max(top_img.width, bottom_img.width)
        
        if top_img.width != target_width:
            new_height = int(top_img.height * (target_width / top_img.width))
            top_img = top_img.resize((target_width, new_height), Image.Resampling.LANCZOS)
            
        if bottom_img.width != target_width:
            new_height = int(bottom_img.height * (target_width / bottom_img.width))
            bottom_img = bottom_img.resize((target_width, new_height), Image.Resampling.LANCZOS)

        # Crear lienzo único combinando ambas prendas
        total_height = top_img.height + bottom_img.height
        combined_outfit = Image.new("RGB", (target_width, total_height), (255, 255, 255))
        
        combined_outfit.paste(top_img, (0, 0))
        combined_outfit.paste(bottom_img, (0, top_img.height))
        
        # Guardar la imagen combinada
        combined_outfit.save(combined_path, "JPEG", quality=95)

        # 3. Conexión a la API de IDM-VTON
        client = Client("yisol/IDM-VTON", token=hf_token)

        # 4. Procesamiento en un solo paso pasando el outfit completo
        res = client.predict(
            dict={
                "background": handle_file(persona_path),
                "layers": [],
                "composite": handle_file(persona_path)
            },
            garm_img=handle_file(combined_path),
            garment_des="overall outfit with top and bottom clothing",
            is_checked=True,
            is_checked_crop=False,
            denoise_steps=30,
            seed=42,
            api_name="/tryon"
        )

        final_path = res[0] if isinstance(res, (list, tuple)) else res

        # 5. Leer y retornar la imagen procesada
        with open(final_path, "rb") as f:
            image_bytes = f.read()

        return Response(content=image_bytes, media_type="image/jpeg")

    except Exception as e:
        print(f"Error en backend: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en procesamiento: {str(e)}")
