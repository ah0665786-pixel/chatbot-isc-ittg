# Chatbot académico ISC – ITTG (núcleo RAG)

Paso 1 del desarrollo: **núcleo RAG probado** (base de conocimiento, recuperación,
generación con respaldo de proveedores, validación y derivación). Aún sin
Telegram ni API web (siguen en los pasos 2 y 3).

## Instalación (Windows, PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip freeze > requirements.lock     # congela versiones que SÍ funcionan
copy .env.example .env             # y llena las API keys
python -m pytest -q                # deben pasar todas
```

## Flujo de trabajo
```powershell
# 1) Pon los PDF/DOCX oficiales en data\docs\ y verifica los modelos del LLM
python -m app.cli check-llm
# 2) Indexa (la 1.ª vez descarga el modelo de embeddings, ~500 MB, una sola vez)
python -m app.cli ingest --version 2026-1
python -m app.cli kb
# 3) Prueba conversando
python -m app.cli chat
# 4) Calibra el umbral con preguntas reales (copia data\golden.example.jsonl -> golden.jsonl)
python -m app.cli calibrate data\golden.jsonl
# 5) Avisos a la coordinadora (ver AVISOS_MODO en .env.example): por defecto solo se
#    guardan en data\outbox\ como .eml. Para probar el envío real a TU correo:
python -m app.cli probar-correo
python -m app.cli ejemplos-avisos  # un .eml de ejemplo de cada tipo y del resumen diario
python -m app.cli avisos           # qué avisos se enviaron, guardaron o fallaron
python -m app.cli accesos          # intentos fallidos de entrar al panel
```
Con `--fake` todo corre sin descargar modelos (solo para probar la plomería).

## Decisiones para que nada se caiga antes de la entrega
| Riesgo | Decisión |
|---|---|
| Retiran el modelo del LLM | Cadena de hasta 3 proveedores en `.env`; modelo retirado = se salta al siguiente y se avisa en el log. `check-llm` lo detecta antes. |
| Cambian/cobran la API de embeddings | Embeddings **locales** (sentence-transformers): sin API, sin costo, sin retiro. |
| Todo lo remoto falla | El 3.er proveedor puede ser **Ollama local**. |
| Mezclar vectores de modelos distintos | El índice guarda con qué modelo se creó y se niega a cargarse con otro. |
| Respuestas inventadas | Solo contesta con fragmentos sobre el umbral; valida cifras/fechas contra el contexto; 2 intentos y deriva. |

## Bot de Telegram (Capa 1)
1. Habla con **@BotFather** en Telegram → `/newbot` → sigue las instrucciones → te da un token.
2. Pégalo en `.env`: `TELEGRAM_BOT_TOKEN=...`
3. Ya con documentos indexados (`python -m app.cli ingest`):
   ```powershell
   python -m app.telegram_bot
   ```
4. Abre el chat con tu bot en Telegram y escríbele. Ctrl+C en la terminal lo detiene.

Es el mismo `ConversationService` que usa `python -m app.cli chat`: misma base de datos, mismas
métricas, mismo pipeline ya probado — Telegram solo es el canal (RNF-PO-03).

## Panel de administración (Capa 1, para la coordinadora)
1. Pon una contraseña en `.env`: `ADMIN_PASSWORD=algo-seguro`
2. Arranca el panel:
   ```powershell
   python -m app.web_admin
   ```
3. Abre `http://127.0.0.1:8000/panel` en el navegador. Pide usuario (`coordinadora` por
   defecto) y la contraseña que pusiste — es autenticación básica del navegador, no hay que
   programar una pantalla de login.
4. Desde ahí: cargar documentos (PDF/DOCX/TXT/MD, se indexan al subirlos) y ver las
   métricas (precisión, derivación, tiempo de respuesta) sin tocar la consola.

Es el mismo `AdminService` que podrías usar por consola; FastAPI solo traduce HTTP, igual
que Telegram solo traduce mensajes — toda la lógica real vive en un solo lugar y ya está
probada aparte.

## Capacidad y costos
Ver `docs/capacidad-y-costos.md`: por qué el plan gratuito de Groq alcanza ~3 consultas/minuto, qué compra el plan Developer (pago por uso, sin suscripción) y el costo real estimado para la población de ISC/ITTG.

## Pendiente (siguientes pasos)
API FastAPI para panel de administración (cargar documentos, ver métricas desde un navegador) ·
Pruebas de carga e inyección de prompts (RNF-EF-02, RNF-SE-04) · Despliegue en AWS.
