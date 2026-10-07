import os
import tempfile
import uuid
import logging
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from gradio_client import Client, handle_file

# Silenciar los logs de red secundarios de Gradio/httpx
logging.getLogger("httpx").setLevel(logging.WARNING)

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
):
    req_id = str(uuid.uuid4())[:8]
    temp_dir = tempfile.gettempdir()

    persona_path = os.path.join(temp_dir, f"{req_id}_person.jpg")
    top_path = os.path.join(temp_dir, f"{req_id}_top.jpg")
    bottom_path = os.path.join(temp_dir, f"{req_id}_bottom.jpg")

    try:
        hf_token = os.getenv("HF_TOKEN")

        # Guardar imágenes de la petición localmente
        with open(persona_path, "wb") as f:
            f.write(await foto_persona.read())
        with open(top_path, "wb") as f:
            f.write(await prenda_top.read())
        with open(bottom_path, "wb") as f:
            f.write(await prenda_bottom.read())

        # Instancia del cliente Gradio
        client = Client("Lacu89/IDM-VTON", token=hf_token)

        # -------------------------------------------------------------
        # PASO 1: Prenda Inferior (BOTTOM) - Se procesa primero
        # -------------------------------------------------------------
        print(f"[{req_id}] Paso 1: Procesando Prenda Inferior (BOTTOM)...")

        res_bottom = client.predict(
            handle_file(persona_path),
            garm_img=handle_file(bottom_path),
            garment_des="a pair of pants, lower body garment, trousers",
            category="lower_body",
            is_checked=True,
            is_checked_crop=False,
            denoise_steps=35,
            seed=42,
            api_name="/tryon",
        )

        bottom_result_path = (
            res_bottom[0] if isinstance(res_bottom, (list, tuple)) else res_bottom
        )

        if not bottom_result_path or not os.path.exists(str(bottom_result_path)):
            raise Exception(f"No se pudo obtener la imagen del BOTTOM: {res_bottom}")

        print(f"[{req_id}] Paso 1 completado exitosamente.")

        # -------------------------------------------------------------
        # PASO 2: Prenda Superior (TOP) - Se procesa sobre la imagen con BOTTOM
        # -------------------------------------------------------------
        print(f"[{req_id}] Paso 2: Procesando Prenda Superior (TOP)...")

        try:
            res_top = client.predict(
                handle_file(bottom_result_path),
                garm_img=handle_file(top_path),
                garment_des="a top clothing, shirt, upper body garment",
                category="upper_body",
                is_checked=True,
                is_checked_crop=False,
                denoise_steps=30,
                seed=42,
                api_name="/tryon",
            )

            final_path = (
                res_top[0] if isinstance(res_top, (list, tuple)) else res_top
            )

            if not final_path or not os.path.exists(str(final_path)):
                raise Exception(f"No se pudo obtener la ruta final: {res_top}")

            print(f"[{req_id}] Paso 2 completado exitosamente.")

            with open(final_path, "rb") as f:
                image_bytes = f.read()

            return Response(content=image_bytes, media_type="image/jpeg")

        except Exception as e_top:
            print(f"[{req_id}] Error en Paso 2 (TOP): {str(e_top)}")
            # Si falla el TOP, devolvemos al menos la imagen con el BOTTOM aplicado
            with open(bottom_result_path, "rb") as f:
                image_bytes = f.read()

            return Response(
                content=image_bytes,
                media_type="image/jpeg",
                headers={
                    "X-VTON-Warning": (
                        "Se devolvió solo el BOTTOM por falla en la prenda superior."
                    )
                },
            )

    except Exception as e:
        print(f"[{req_id}] Error general: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error en el servidor: {str(e)}"
        )

    finally:
        # Limpieza de archivos temporales
        for path in [persona_path, top_path, bottom_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
