import os
import uuid
import tempfile
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
    req_id = str(uuid.uuid4())[:8]
    temp_dir = tempfile.gettempdir()
    
    persona_path = os.path.join(temp_dir, f"{req_id}_person.jpg")
    top_path = os.path.join(temp_dir, f"{req_id}_top.jpg")
    bottom_path = os.path.join(temp_dir, f"{req_id}_bottom.jpg")

    try:
        hf_token = os.getenv("HF_TOKEN")
        
        with open(persona_path, "wb") as f:
            f.write(await foto_persona.read())
        with open(top_path, "wb") as f:
            f.write(await prenda_top.read())
        with open(bottom_path, "wb") as f:
            f.write(await prenda_bottom.read())

        client = Client("Lacu89/IDM-VTON", token=hf_token)

        # -------------------------------------------------------------
        # PASO 1: Prenda Superior (TOP)
        # -------------------------------------------------------------
        print(f"[{req_id}] Paso 1: Procesando TOP...")

        res_top = client.predict(
            dict={
                "background": handle_file(persona_path),
                "layers": [],
                "composite": handle_file(persona_path)
            },
            garm_img=handle_file(top_path),
            garment_des="top clothing",
            category="upper_body",
            is_checked=True,
            is_checked_crop=False,
            denoise_steps=30,
            seed=42,
            api_name="/tryon"
        )

        top_result_path = res_top[0] if isinstance(res_top, (list, tuple)) else res_top

        # -------------------------------------------------------------
        # PASO 2: Prenda Inferior (BOTTOM)
        # -------------------------------------------------------------
        print(f"[{req_id}] Paso 2: Procesando BOTTOM (Protegiendo calzado y entrepierna)...")
        
        try:
            res_bottom = client.predict(
                dict={
                    "background": handle_file(top_result_path),
                    "layers": [],
                    "composite": handle_file(top_result_path)
                },
                garm_img=handle_file(bottom_path),
                garment_des="pants, preserve shoes and footwear",
                category="dresses",          # Cambiado a 'dresses' para conservar la segmentación de piernas/zapatos
                is_checked=True,
                is_checked_crop=False,
                denoise_steps=20,            # Fuerza a mantener la estructura base (zapatos/entrepierna)
                seed=42,
                api_name="/tryon"
            )

            final_path = res_bottom[0] if isinstance(res_bottom, (list, tuple)) else res_bottom

            if not final_path or not os.path.exists(str(final_path)):
                raise Exception("Error al obtener la ruta de la imagen generada.")

            with open(final_path, "rb") as f:
                image_bytes = f.read()

            return Response(content=image_bytes, media_type="image/jpeg")

        except Exception as e_bottom:
            print(f"[{req_id}] Fallo en Paso 2: {str(e_bottom)}")
            with open(top_result_path, "rb") as f:
                image_bytes = f.read()
                
            return Response(
                content=image_bytes, 
                media_type="image/jpeg", 
                headers={"X-VTON-Warning": "Se devolvió resultado parcial del Paso 1."}
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en servidor: {str(e)}")

    finally:
        for path in [persona_path, top_path, bottom_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
