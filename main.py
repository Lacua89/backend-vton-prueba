import os
from fastapi import FastAPI, File, UploadFile, HTTPException
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
    prenda_bottom: UploadFile = File(...)
):
    try:
        hf_token = os.getenv("HF_TOKEN")
        
        persona_path = "temp_person.jpg"
        top_path = "temp_top.jpg"
        
        with open(persona_path, "wb") as f:
            f.write(await foto_persona.read())
        with open(top_path, "wb") as f:
            f.write(await prenda_top.read())

        # Conexión al Space Kwai-Kolors
        client = Client("Kwai-Kolors/Kolors-Virtual-Try-On", token=hf_token)

        # Usamos fn_index=1 que corresponde al botón de procesamiento principal
        result = client.predict(
            person_img=handle_file(persona_path),
            garment_img=handle_file(top_path),
            seed=0,
            randomize_seed=True,
            fn_index=1
        )

        print(f"Respuesta cruda de Gradio: {result}")

        # Extracción de la ruta de la imagen
        image_path = None

        if isinstance(result, (list, tuple)) and len(result) > 0:
            item = result[0]
            if isinstance(item, dict) and "image" in item:
                image_path = item["image"]
            elif isinstance(item, str):
                image_path = item
        elif isinstance(result, dict) and "image" in result:
            image_path = result["image"]
        elif isinstance(result, str):
            image_path = result

        if not image_path or not os.path.exists(str(image_path)):
            raise Exception(f"No se pudo obtener la imagen del modelo. Respuesta: {result}")

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        return Response(content=image_bytes, media_type="image/jpeg")

    except Exception as e:
        print(f"Error en backend: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en procesamiento: {str(e)}")
