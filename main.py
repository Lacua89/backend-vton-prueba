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
    return {"status": "ok", "message": "API VTON Activa (IDM-VTON Full Outfit via Lacu89/IDM-VTON)"}

@app.post("/api/v1/try-on-completo")
async def try_on(
    foto_persona: UploadFile = File(...),
    prenda_top: UploadFile = File(...),
    prenda_bottom: UploadFile = File(...)
):
    # Generar prefijo único para evitar colisiones entre peticiones concurrentes
    req_id = str(uuid.uuid4())[:8]
    temp_dir = tempfile.gettempdir()
    
    persona_path = os.path.join(temp_dir, f"{req_id}_person.jpg")
    top_path = os.path.join(temp_dir, f"{req_id}_top.jpg")
    bottom_path = os.path.join(temp_dir, f"{req_id}_bottom.jpg")

    try:
        hf_token = os.getenv("HF_TOKEN")
        
        # 1. Guardar archivos locales temporales
        with open(persona_path, "wb") as f:
            f.write(await foto_persona.read())
        with open(top_path, "wb") as f:
            f.write(await prenda_top.read())
        with open(bottom_path, "wb") as f:
            f.write(await prenda_bottom.read())

        # Conexión al Space de IDM-VTON
        client = Client("Lacu89/IDM-VTON", token=hf_token)

        # -------------------------------------------------------------
        # PASO 1: Procesar Prenda Superior (TOP)
        # -------------------------------------------------------------
        print(f"[{req_id}] Paso 1: Procesando Prenda Superior (TOP)...")

        res_top = client.predict(
            dict={
                "background": handle_file(persona_path),
                "layers": [],
                "composite": handle_file(persona_path)
            },
            garm_img=handle_file(top_path),
            garment_des="high quality upper body clothing, t-shirt or shirt",
            category="upper_body",
            is_checked=True,
            is_checked_crop=False,      # Mantiene cuerpo entero para permitir el paso de pantalones
            denoise_steps=30,
            seed=42,
            api_name="/tryon"
        )

        top_result_path = res_top[0] if isinstance(res_top, (list, tuple)) else res_top
        print(f"[{req_id}] Paso 1 completado: {top_result_path}")

        # -------------------------------------------------------------
        # PASO 2: Procesar Prenda Inferior (BOTTOM)
        # -------------------------------------------------------------
        print(f"[{req_id}] Paso 2: Procesando Prenda Inferior (BOTTOM)...")
        
        try:
            res_bottom = client.predict(
                dict={
                    "background": handle_file(top_result_path),  # Imagen previa con el TOP ya aplicado
                    "layers": [],
                    "composite": handle_file(top_result_path)
                },
                garm_img=handle_file(bottom_path),
                # Prompt enriquecido para guiar mejor el ajuste de pantalones/faldas
                garment_des="high quality detailed pants, perfect fit, detailed fabric texture",
                category="lower_body",
                is_checked=True,
                is_checked_crop=False,
                denoise_steps=35,        # Subido a 35 para mejorar texturas y bordes del pantalón
                seed=42,
                api_name="/tryon"
            )

            final_path = res_bottom[0] if isinstance(res_bottom, (list, tuple)) else res_bottom

            if not final_path or not os.path.exists(str(final_path)):
                raise Exception(f"No se pudo resolver la ruta final en IDM-VTON: {res_bottom}")

            print(f"[{req_id}] Paso 2 completado exitosamente: {final_path}")

            with open(final_path, "rb") as f:
                image_bytes = f.read()

            return Response(content=image_bytes, media_type="image/jpeg")

        except Exception as e_bottom:
            print(f"[{req_id}] Error en Paso 2 (Prenda Inferior): {str(e_bottom)}")
            print(f"[{req_id}] Devolviendo resultado parcial (Paso 1).")
            
            with open(top_result_path, "rb") as f:
                image_bytes = f.read()
                
            return Response(
                content=image_bytes, 
                media_type="image/jpeg", 
                headers={"X-VTON-Warning": "Solo se procesó la prenda superior debido a un error en la prenda inferior."}
            )

    except Exception as e:
        print(f"[{req_id}] Error crítico en backend: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en procesamiento: {str(e)}")

    finally:
        # Limpieza de archivos temporales subidos por el cliente
        for path in [persona_path, top_path, bottom_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
