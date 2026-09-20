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
    return {"status": "ok", "message": "API VTON Gratis Activa (IDM Top Crop -> OOT Bottom LongFix)"}

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
        
        # 1. Guardar archivos locales temporales
        with open(persona_path, "wb") as f:
            f.write(await foto_persona.read())
        with open(top_path, "wb") as f:
            f.write(await prenda_top.read())
        with open(bottom_path, "wb") as f:
            f.write(await prenda_bottom.read())

        # -------------------------------------------------------------
        # PASO 1: Procesar Prenda Superior (Top) con IDM-VTON + Crop
        # -------------------------------------------------------------
        print("Iniciando Paso 1: Procesando Prenda Superior con IDM-VTON...")
        client_top = Client("yisol/IDM-VTON", token=hf_token)

        res_top = client_top.predict(
            dict={
                "background": handle_file(persona_path),
                "layers": [],
                "composite": handle_file(persona_path)
            },
            garm_img=handle_file(top_path),
            garment_des="upper body clothing",
            is_checked=True,
            is_checked_crop=True, # Mantenemos el crop para limpiar el fondo
            denoise_steps=30,
            seed=42,
            api_name="/tryon"
        )

        top_result_path = res_top[0] if isinstance(res_top, (list, tuple)) else res_top
        print(f"Paso 1 completado: {top_result_path}")

        # -------------------------------------------------------------
        # PASO 2: Procesar Prenda Inferior (Bottom) con OOTDiffusion (Optimizado para Largo)
        # Volvemos a OOTDiffusion porque Cat-VTON Spaces estan caidos, pero con tweaks.
        # -------------------------------------------------------------
        print("Iniciando Paso 2: Procesando Prenda Inferior con OOTDiffusion (LongFix)...")
        
        try:
            client_bottom = Client("levihsu/OOTDiffusion", token=hf_token)

            res_bottom = client_bottom.predict(
                vton_img=handle_file(top_result_path), # Usamos el resultado del Paso 1
                garm_img=handle_file(bottom_path),
                category="Lower-body",
                n_samples=1,
                n_steps=35,        # Pasos altos para calidad
                image_scale=1.5,   # <--- CLAVE: Bajamos la escala (de 3.0 a 1.5). 
                                  # Esto le da mas libertad a la IA para extender el pantalon hacia abajo.
                seed=42,
                api_name="/process_dc"
            )

            # Extraer ruta física
            final_path = None
            if isinstance(res_bottom, (list, tuple)) and len(res_bottom) > 0:
                item = res_bottom[0]
                if isinstance(item, dict):
                    final_path = item.get("image") or item.get("name") or item.get("path")
                elif isinstance(item, str):
                    final_path = item
            elif isinstance(res_bottom, dict):
                final_path = res_bottom.get("image") or res_bottom.get("name") or res_bottom.get("path")
            elif isinstance(res_bottom, str):
                final_path = res_bottom

            if not final_path or not os.path.exists(str(final_path)):
                raise Exception(f"No se pudo resolver la ruta final en OOTDiffusion: {res_bottom}")

            print(f"Paso 2 completado exitosamente: {final_path}")

            # 3. Responder con la imagen final combinada
            with open(final_path, "rb") as f:
                image_bytes = f.read()

            return Response(content=image_bytes, media_type="image/jpeg")

        except Exception as e_bottom:
            print(f"Error en Paso 2 (OOTDiffusion Optimizado): {str(e_bottom)}")
            # Mantenemos la contingencia: si falla el pantalon, devolvemos al menos la remera.
            print("Devolviendo resultado parcial del Paso 1 debido a error en Paso 2.")
            with open(top_result_path, "rb") as f:
                image_bytes = f.read()
            return Response(content=image_bytes, media_type="image/jpeg", headers={"X-VTON-Warning": "Solo se proceso la prenda superior debido a un error técnico en la prenda inferior."})

    except Exception as e:
        print(f"Error crítico en backend: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en procesamiento: {str(e)}")
