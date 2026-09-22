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
    return {"status": "ok", "message": "API VTON Activa (IDM-VTON Full Outfit via Lacu89/IDM-VTON)"}

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

        # Conexión única a tu Space
        client = Client("Lacu89/IDM-VTON", token=hf_token)

        # -------------------------------------------------------------
        # PASO 1: Procesar Prenda Superior (TOP) con tu Space IDM-VTON
        # -------------------------------------------------------------
        print("Iniciando Paso 1: Procesando Prenda Superior (TOP) con IDM-VTON...")

        res_top = client.predict(
            dict={
                "background": handle_file(persona_path),
                "layers": [],
                "composite": handle_file(persona_path)
            },
            garm_img=handle_file(top_path),
            garment_des="upper body clothing",
            category="upper_body",      # <--- Categoría TOP
            is_checked=True,
            is_checked_crop=False,      # Recomendado en False para no recortar piernas antes del Paso 2
            denoise_steps=30,
            seed=42,
            api_name="/tryon"
        )

        top_result_path = res_top[0] if isinstance(res_top, (list, tuple)) else res_top
        print(f"Paso 1 completado: {top_result_path}")

        # -------------------------------------------------------------
        # PASO 2: Procesar Prenda Inferior (BOTTOM) con tu Space IDM-VTON
        # -------------------------------------------------------------
        print("Iniciando Paso 2: Procesando Prenda Inferior (BOTTOM) con IDM-VTON...")
        
        try:
            res_bottom = client.predict(
                dict={
                    "background": handle_file(top_result_path),  # Usamos la persona con el TOP ya puesto
                    "layers": [],
                    "composite": handle_file(top_result_path)
                },
                garm_img=handle_file(bottom_path),
                garment_des="lower body clothing",
                category="lower_body",  # <--- Categoría BOTTOM
                is_checked=True,
                is_checked_crop=False,
                denoise_steps=30,
                seed=42,
                api_name="/tryon"
            )

            # Extraer ruta física del resultado
            final_path = res_bottom[0] if isinstance(res_bottom, (list, tuple)) else res_bottom

            if not final_path or not os.path.exists(str(final_path)):
                raise Exception(f"No se pudo resolver la ruta final en IDM-VTON: {res_bottom}")

            print(f"Paso 2 completado exitosamente: {final_path}")

            # Responder con la imagen final combinada
            with open(final_path, "rb") as f:
                image_bytes = f.read()

            return Response(content=image_bytes, media_type="image/jpeg")

        except Exception as e_bottom:
            print(f"Error en Paso 2 (Prenda Inferior): {str(e_bottom)}")
            # Contingencia: si falla la prenda inferior, devolvemos el resultado de la remera
            print("Devolviendo resultado parcial del Paso 1 debido a error en Paso 2.")
            with open(top_result_path, "rb") as f:
                image_bytes = f.read()
            return Response(
                content=image_bytes, 
                media_type="image/jpeg", 
                headers={"X-VTON-Warning": "Solo se procesó la prenda superior debido a un error técnico en la prenda inferior."}
            )

    except Exception as e:
        print(f"Error crítico en backend: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en procesamiento: {str(e)}")
