import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from gradio_client import Client, handle_file

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
    prenda_bottom: UploadFile = File(...),
    # Opcionales para talle y ajustes
    categoria: str = Form("tops"),       # 'tops', 'bottoms' o 'dresses'
    talle_largo: bool = Form(False)      # True si es prenda holgada/larga
):
    try:
        token = os.getenv("HF_TOKEN")
        
        persona_path = "temp_person.jpg"
        top_path = "temp_top.jpg"
        
        with open(persona_path, "wb") as f:
            f.write(await foto_persona.read())
        with open(top_path, "wb") as f:
            f.write(await prenda_top.read())

        client = Client("fashn-ai/fashn-vton-1-5", hf_token=token)

        # Llamada con protección de rostro y parámetros de talle
        result = client.predict(
            model_image=handle_file(persona_path),
            garment_image=handle_file(top_path),
            category=categoria,          # Control de talle/tipo
            nsfw_filter=True,
            cover_feet=False,
            adjust_hands=True,           # Ajuste de manos frente a la prenda
            restore_background=True,     # PROTECCIÓN DE ROSTRO Y FONDO ORIGINAL
            restore_clothes=False,
            long_top=talle_largo,        # Ajuste para prendas largas/holgadas
            guidance_scale=2.5,
            timesteps=30,
            seed=42,
            num_samples=1,
            api_name="/process_tryon"
        )

        image_path = result[0]['image'] if isinstance(result, list) else result

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        return Response(content=image_bytes, media_type="image/jpeg")

    except Exception as e:
        print(f"Error en backend: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en procesamiento: {str(e)}")
