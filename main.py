import os
import uuid
import tempfile
import logging
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from gradio_client import Client, handle_file

# Configuración de logs para monitorear el flujo de peticiones
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vton-api")

app = FastAPI(title="API VTON", version="1.0.0")

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
    # Validar la presencia del token antes de procesar archivos
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.error("No se encontró la variable de entorno HF_TOKEN.")
        raise HTTPException(
            status_code=500,
            detail="Error de configuración en el servidor: HF_TOKEN no está definido."
        )

    req_id = str(uuid.uuid4())[:8]
    temp_dir = tempfile.gettempdir()

    # Definir rutas temporales únicas
    persona_path = os.path.join(temp_dir, f"{req_id}_person.jpg")
    top_path = os.path.join(temp_dir, f"{req_id}_top.jpg")
    bottom_path = os.path.join(temp_dir, f"{req_id}_bottom.jpg")

    # Lista para llevar registro de todos los archivos generados y poder limpiarlos en el 'finally'
    paths_to_clean: List[str] = [persona_path, top_path, bottom_path]

    try:
        # 1. Guardar archivos cargados en el disco
        with open(persona_path, "wb") as f:
            f.write(await foto_persona.read())
        with open(top_path, "wb") as f:
            f.write(await prenda_top.read())
        with open(bottom_path, "wb") as f:
            f.write(await prenda_bottom.read())

        # Instanciar el cliente Gradio
        client = Client("Lacu89/IDM-VTON", token=hf_token)

        # -------------------------------------------------------------
        # PASO 1: Procesar Prenda Superior (TOP)
        # -------------------------------------------------------------
        logger.info(f"[{req_id}] Paso 1: Procesando TOP...")

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

        if not top_result_path or not os.path.exists(str(top_result_path)):
            raise Exception("No se pudo obtener el resultado del Paso 1 (TOP).")

        # Registramos el resultado del paso 1 para limpiarlo al finalizar
        paths_to_clean.append(str(top_result_path))

        # -------------------------------------------------------------
        # PASO 2: Procesar Prenda Inferior (BOTTOM)
        # -------------------------------------------------------------
        logger.info(f"[{req_id}] Paso 2: Procesando BOTTOM...")

        try:
            res_bottom = client.predict(
                dict={
                    "background": handle_file(top_result_path),
                    "layers": [],
                    "composite": handle_file(top_result_path)
                },
                garm_img=handle_file(bottom_path),
                garment_des="pants, lower body garment, preserve footwear",
                category="lower_body",  # Usar 'lower_body' para la prenda inferior (o 'dresses' si el pipeline específico lo requiere)
                is_checked=True,
                is_checked_crop=False,
                denoise_steps=20,
                seed=42,
                api_name="/tryon"
            )

            final_path = res_bottom[0] if isinstance(res_bottom, (list, tuple)) else res_bottom

            if not final_path or not os.path.exists(str(final_path)):
                raise Exception("No se pudo obtener el resultado final del Paso 2 (BOTTOM).")

            paths_to_clean.append(str(final_path))

            with open(final_path, "rb") as f:
                image_bytes = f.read()

            return Response(content=image_bytes, media_type="image/jpeg")

        except Exception as e_bottom:
            logger.warning(f"[{req_id}] Falló el Paso 2: {str(e_bottom)}. Retornando resultado parcial (TOP).")

            with open(top_result_path, "rb") as f:
                image_bytes = f.read()

            return Response(
                content=image_bytes,
                media_type="image/jpeg",
                headers={"X-VTON-Warning": "Se devolvió el resultado parcial del Paso 1."}
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{req_id}] Error no controlado: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

    finally:
        # Garantizar la eliminación de todos los archivos creados
        for path in paths_to_clean:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as cleanup_err:
                    logger.warning(f"[{req_id}] No se pudo eliminar el archivo temporal '{path}': {cleanup_err}")
