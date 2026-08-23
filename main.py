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
        bottom_path = "temp_bottom.jpg"
        result_top_path = "temp_result_top.jpg"
        
        # Guardar archivos recibidos desde Flutter
        with open(persona_path, "wb") as f:
            f.write(await foto_persona.read())
        with open(top_path, "wb") as f:
            f.write(await prenda_top.read())
        with open(bottom_path, "wb") as f:
            f.write(await prenda_bottom.read())

        client = Client("yisol/IDM-VTON", token=hf_token)

        # --- PASO 1: Vestir la prenda superior (Top) ---
        res_top = client.predict(
            dict={
                "background": handle_file(persona_path),
                "layers": [],
                "composite": handle_file(persona_path)
            },
            garm_img=handle_file(top_path),
            garment_des="upper body clothing",
            is_checked=True,
            is_checked_crop=False,
            denoise_steps=30,
            seed=42,
            api_name="/tryon"
        )

        out_top_path = res_top[0] if isinstance(res_top, (list, tuple)) else res_top

        # --- PASO 2: Vestir la prenda inferior (Bottom) sobre el resultado del Top ---
        res_bottom = client.predict(
            dict={
                "background": handle_file(out_top_path),
                "layers": [],
                "composite": handle_file(out_top_path)
            },
            garm_img=handle_file(bottom_path),
            garment_des="lower body clothing",
            is_checked=True,
            is_checked_crop=False,
            denoise_steps=30,
            seed=42,
            api_name="/tryon"
        )

        final_path = res_bottom[0] if isinstance(res_bottom, (list, tuple)) else res_bottom

        # Leer y devolver la imagen final combinada
        with open(final_path, "rb") as f:
            image_bytes = f.read()

        return Response(content=image_bytes, media_type="image/jpeg")

    except Exception as e:
        print(f"Error en backend: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en procesamiento: {str(e)}")
