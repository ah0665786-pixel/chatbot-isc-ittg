# Matriz de trazabilidad — Chatbot académico ISC (ITTG)

Auditoría del 2026-10-24 sobre el commit `f1345b7` (rama `main`), contra
`docs/protocolo_requisitos.md` (protocolo 2601-11: RF, RNF, HU de la **Lista B**, Tablas 60-74, y UC).

Leyenda: ✅ cumple y tiene prueba o evidencia · 🟡 parcial, sin prueba o sin medición · ❌ no implementado.

## Resumen

| Grupo | Total | ✅ | 🟡 | ❌ |
|---|---|---|---|---|
| RF-I (información) | 5 | 5 | 0 | 0 |
| RF-C (comportamiento) | 6 | 4 | 1 | 1 |
| RF-RN (reglas de negocio) | 3 | 2 | 1 | 0 |
| RNF | 17 | 5 | 11 | 1 |
| Criterios de aceptación HU-01 a HU-15 | 61 | 52 | 8 | 1 |

Los RNF casi nunca están en ❌ porque la funcionalidad existe; lo que falta es el despliegue
en AWS o la **medición** que pide el criterio (encuestas, pruebas de integración, respaldos,
cifrado, monitoreo).

## 1. Huecos

### 1.1 Huecos nuevos (aparecen al auditar contra la lista oficial)

| # | Hueco | Requisito | Verificación |
|---|---|---|---|
| N1 | ✅ **Resuelto el 2026-10-24** (ver abajo). **Las preguntas de seguimiento buscaban sin contexto**: el historial va al LLM, pero la búsqueda usa solo la pregunta actual. Tras "¿Cuántas horas dura el servicio social?", la pregunta "¿Y cuántas horas son?" recupera 0 de 5 fragmentos útiles (el 1.º es el cuestionario de hábitos de estudio) y aun así supera el umbral (0.858). **Corrección**: si la pregunta es de seguimiento (empieza con "y", "entonces"..., dice "eso", "ahí"... o no nombra ningún tema), la búsqueda antepone las preguntas anteriores del estudiante hasta la última que sí nombraba el tema (`search_query` en `app/rag/query_expansion.py`), sin llamar al LLM; el LLM sigue viendo la pregunta original. Medido en el contenedor con 7 seguimientos reales: antes 2/7 con el fragmento correcto en el top 5, ahora 7/7 ("¿Y cuántas horas son?": 3 de 5 útiles, 1.º el correcto, 0.887). Ninguna de las 62 preguntas de `data/retrieval_cases.jsonl` se toma como seguimiento; con una pregunta de otro tema en el historial siguen 54/54, y `calibrate` sigue en Hit@5 = 100 %. | HU-04 CA-02 | Medido en el contenedor con el índice real; `test_pipeline.py` (12 pruebas de seguimiento) |
| N2 | ✅ **Resuelto el 2026-10-24**. **Editar un mensaje hacía fallar al bot**: Telegram reenvía el mensaje editado, `on_message` lo atiende, pero `update.message` es `None` (llega en `edited_message`) y la función lanzaba `AttributeError` antes del `try`: el estudiante no recibía respuesta. **Corrección**: `on_message` y `start` usan `update.effective_message`, así que se responde la versión corregida de la pregunta (se prefirió a ignorar las ediciones con `filters.UpdateType.MESSAGE`, porque el estudiante que corrige una errata espera respuesta). | RF-I-01, HU-02 CA-01 | `test_telegram_bot.py` (3, con `Update.de_json` simulado, sin red): fallaban 2 antes del cambio |
| N3 | **Telegram no pasa por una API REST**: el protocolo pide que el canal se comunique con el backend por FastAPI. Hoy `app/telegram_bot.py` construye en el mismo proceso el `ConversationService` (BD + motor de IA). La capa de interfaz no tiene lógica de negocio, pero sí acceso directo. | RNF-PO-03 | Lectura de código |
| N4 | **Sin despliegue 24/7 ni monitoreo**: todo corre en Docker en la máquina de desarrollo; no hay EC2, monitoreo de disponibilidad ni reinicio fuera de `restart: unless-stopped`. | RF-C-05, RNF-FI-01, RNF-PO-01, HU-12 CA-03 | — |
| N5 | **Sin respaldos**: el volumen `pgdata` no se respalda (el protocolo pide respaldo diario en S3). | RNF-FI-03 | — |
| N6 | **Sin cifrado en reposo** de la base (el protocolo lo pide vía RDS); se guardan `telegram_id`, nombre de Telegram y el texto libre de las preguntas. | RNF-SE-02 | — |
| N7 | **El panel no usa HTTPS**: hoy solo escucha en `127.0.0.1`; para exponerlo hace falta un proxy con TLS. | RNF-SE-03 | `docker-compose.yml` |
| N8 | **Textos de ISC fijos en el código** (11 lugares: instrucciones del LLM, bienvenida, mensajes de derivación, asunto del correo, panel): usar el sistema en otra carrera exige editar código, no solo configuración. | RNF-PO-02 | `grep` en `app/` |
| N9 | ✅ **Resuelto el 2026-10-26** (rama `reportes-coordinadora`). **No existían "tipos de consulta más frecuentes" ni "áreas con más derivación"**: no se clasificaba el tema de cada consulta; solo había motivos de derivación. **Corrección**: cada consulta se guarda con uno de 9 temas (plan de estudios, residencia profesional, servicio social, titulación, bajas, traslados, actividades complementarias, permisos, otros) en la columna nueva `metricas.tema` (`classify_topic`, ver N11). Las consultas anteriores quedan sin tema y se muestran como "Sin clasificar" (no se reclasifican). El panel tiene una sección "Reportes" con los tipos de consulta más frecuentes (consultas y % del total por tema) y los temas con más derivaciones (derivadas y tasa de derivación del tema); el CSV por consulta tiene la columna "Tema". El reporte se calcula en `app/reports.py` sobre `metric_rows`, que ya comparten SQLite y PostgreSQL. | HU-13 CA-01, HU-15 CA-03 | `test_reports.py::test_tipos_mas_frecuentes_y_temas_con_mas_derivaciones`, `test_metrics_export.py::test_csv_columnas_etiquetas_y_hora_local`; `test_repository_postgres.py::test_tema_reporte_y_p95` en Docker |
| N10 | ✅ **Resuelto el 2026-10-26** (rama `reportes-coordinadora`). **Desempeño histórico incompleto**: no había agrupación por semana/mes/ciclo, ni aviso de "periodo mínimo" cuando faltan datos, ni "último reporte" cuando no hay datos en el periodo. **Corrección** (`app/reports.py`, sección "Reportes" del panel y `/reportes/exportar.csv`): filtro por fechas en hora de Tuxtla (ambos días incluidos; campos vacíos = sin fecha, como en `/metricas/exportar.csv`) y agrupación por semana (lunes a domingo), mes o ciclo escolar. El ciclo del TecNM se define solo en `SCHOOL_CYCLES`: Enero-Junio (1 ene-30 jun), Verano (julio) y Agosto-Diciembre (1 ago-31 dic), numerados 1, 2 y 3 como en el SII. "Suficiente" = `REPORTES_MIN_CONSULTAS` (20 por defecto): con menos de 20 tiempos el p95 es simplemente el máximo. Con menos consultas el panel dice cuántas hay y el periodo mínimo necesario ("del 04/10/2026 al 25/10/2026": desde el día en que se juntan las 20 más recientes) y marca cada periodo con "pocos datos"; si el periodo no tiene ninguna consulta, muestra el último reporte disponible ("Último reporte disponible: Octubre 2026 (17 consultas)"). **Medido en Docker** con la base real (17 consultas, todas "Sin clasificar"): "Datos insuficientes: en total hay 17 consultas registradas (desde el 22/10/2026) y se necesitan al menos 20". | HU-15 CA-02, CA-04; HU-09 CA-04 | `test_reports.py` (34: periodos, cortes a medianoche de Tuxtla en semana/mes/ciclo, periodo mínimo, sin datos, panel y CSV) |
| N11 | 🟡 **Avanzado el 2026-10-26** (rama `reportes-coordinadora`). **Clasificación de intención solo parcial**: la expansión de consulta reconocía titulación, convalidación, repetición y actividades complementarias, pero no inscripción, baja, residencia ni servicio social, y no extrae entidades. **Corrección**: `classify_topic` (`app/rag/query_expansion.py`) reutiliza los patrones de la expansión (`_TITULACION`, `_CONVALIDACION`, `_REPETICION`, `_ACTIVIDADES`) y agrega residencia, servicio social, bajas, traslados, permisos y plan de estudios (inscripción, reinscripción, carga y repetición de materias van aquí), con prioridad al trámite que se quiere hacer ("¿Me puedo titular con la residencia?" es titulación). En una pregunta de seguimiento sin tema propio ("¿Y cuántas horas son?") el tema sale de las preguntas previas del estudiante, igual que en `search_query`; si nombra el suyo ("¿Y la residencia?"), gana el suyo. **Medido** con las 62 preguntas de `data/retrieval_cases.jsonl` (se completó el campo `tema` en las 23 que no lo tenían): 62/62 (titulación 20/20, residencia 8/8, otros 8/8, actividades 7/7, servicio social 6/6, traslados 6/6, plan de estudios 6/6, bajas 1/1). **Falta**: el conjunto no trae preguntas de permisos ni de traslado propiamente dicho (solo convalidación), así que esos temas solo tienen las pruebas escritas a mano; no hay extracción de entidades. | RF-C-01, HU-05 CA-01/CA-02 | `test_topics.py` (27, con la medición por tema y 7 casos de seguimiento) |
| N12 | **Criterios de usabilidad sin medir**: 90 % de usuarios nuevos sin instrucciones, 85 % con calificación ≥ 4 (no se calcula en ningún reporte), prueba en Android/iOS/web/escritorio, carga de 50 páginas en ≤ 10 min. | RNF-US-01 a 04 | — |
| N13 | ✅ **Resuelto el 2026-10-26** (rama `reportes-coordinadora`). **El sistema no reportaba el p95**: el criterio es "95 % ≤ 30 s, medido en METRICAS"; se había medido a mano (1.69 s en 10 consultas). **Corrección**: `percentile` (`app/reports.py`, rango más cercano) en `metrics_summary` de SQLite y PostgreSQL (tarjeta "Tiempo p95 (meta: ≤ 30 s)" del panel y `/api/metricas`), en el reporte por periodo y en el CSV del reporte (columnas "Tiempo promedio (s)" y "Tiempo p95 (s)" por periodo y total). **Medido en Docker** con la base real: 17 consultas, promedio 0.99 s, p95 2.00 s. | RNF-EF-01 | `test_reports.py::test_p95_por_rango_mas_cercano`, `test_resumen_trae_p95`, `test_csv_del_reporte`; `test_repository_postgres.py::test_tema_reporte_y_p95` |
| N14 | **La numeración de HU en el código es la Lista A**: p. ej. el indicador de escritura dice "HU-01 CA-03" y es HU-02 CA-03 en la Lista B (ver sección 2). | Todas | — |
| N15 | ✅ **Resuelto el 2026-10-24**. **El bot no arrancaba si el panel había leído métricas**: con `autocommit = False`, cada lectura de `PostgresRepository` (`metrics_summary`, `get_history`) abría una transacción que quedaba viva hasta la siguiente escritura, y el panel casi nunca escribe; su sesión quedó "idle in transaction" casi 4 min. Al reconstruir el bot, el `ALTER TABLE metricas ADD COLUMN IF NOT EXISTS` del esquema pedía un candado ACCESS EXCLUSIVE (aunque la columna ya existiera) y esperó para siempre (`wait_event_type = Lock`); solo se liberó reiniciando el panel. **Corrección** (`app/adapters/db_postgres.py`): toda lectura pasa por `_tx` y cierra su transacción; el ALTER se ejecuta solo si `information_schema.columns` dice que falta la columna (lista `MIGRATIONS`, dentro del mismo candado consultivo); y la creación del esquema usa `lock_timeout = 10s`, así un bloqueo futuro falla con "lock timeout" en el log en vez de colgar el arranque. Medido en Docker con el mismo procedimiento (leer el panel y reconstruir el bot): con `main` (`1a3e368`) la sesión del panel quedó "idle in transaction" y el bot siguió sin "Application started" a los 106 s, con su `CREATE TABLE` esperando un candado (`Lock`) hasta que se reinició el panel; con la corrección la sesión queda "idle" y el bot arranca en 14 s. | RNF-FI-01, RNF-FI-03 | `test_repository_postgres.py` (4 nuevas, en Docker contra una base de prueba aparte): contra `main` fallan 3, una con "el segundo repositorio se quedó esperando el candado (más de 15 s)"; con la corrección pasan las 8 |

### 1.2 Huecos de la auditoría anterior que siguen abiertos (ahora con su número oficial)

| Hueco | Requisito oficial |
|---|---|
| ✅ **Resuelto el 2026-10-24** (panel). Accesos no autorizados sin registro: el panel respondía 401 y solo quedaba la línea genérica de uvicorn (IP y ruta, sin usuario, nada en la base). **Corrección**: `require_auth` (`app/web_admin.py`) registra cada intento con credenciales incorrectas (usuario intentado, recortado a 100 caracteres; IP; ruta; fecha y hora UTC) en el log (`WARNING web_admin: acceso no autorizado al panel: usuario='intruso' ip=172.19.0.1 ruta=/panel`) y en la tabla nueva `accesos_fallidos`; la contraseña intentada no se guarda en ningún lado. Una petición sin credenciales (el navegador pidiendo usuario) no cuenta como intento. De paso, la comparación se hace en bytes: con un `ADMIN_USER` con acento, `secrets.compare_digest` lanzaba `TypeError` (error 500) en cada intento. Consulta: `python -m app.cli accesos`. **Medido en Docker** (código de la rama, base real): antes, del intento `intruso` con contraseña incorrecta no quedaba nada en la base; después, 1 fila en `accesos_fallidos` (`intruso`, `172.19.0.1`, `/panel`) y la contraseña intentada aparece 0 veces en el log y en la tabla. En Docker la IP es la de la red de Docker, no la del equipo. Pruebas: `test_web_admin.py` (5 nuevas), `test_repository.py::test_accesos_fallidos`. **Sigue abierto** para Telegram: con registro automático por ID no existe un "usuario no registrado" (RNF-SE-01) | **HU-01 CA-03**, RNF-SE-01 |
| ✅ **Resuelto el 2026-10-24** (falta la prueba de entrega real, ver abajo). Aviso por correo a un buzón de pruebas (`sandbox.smtp.mailtrap.io`), sin registro de envíos, síncrono, sin variables en `.env.example`. **Corrección** (`app/notify.py`): `AVISOS_MODO` con 4 modos: `vista_previa` (por defecto: guarda cada aviso como `.eml` legible en `data/outbox/`, fuera de git, sin enviar), `prueba` (envía solo a `AVISOS_CORREO_PRUEBA`; si está vacío NO envía, guarda y advierte en el log), `coordinadora` (envía a `COORDINADORA_EMAIL`; solo con ese valor escrito a mano) y `desactivado` (lo usa la prueba de carga, que antes habría avisado por cada consulta derivada). Gmail por SMTP (`smtp.gmail.com:587`, STARTTLS, contraseña de aplicación) con `From` = la cuenta autenticada `tellbotittg@gmail.com`. El aviso corre en un hilo aparte (`Notifier`) después de registrar la métrica, y cada resultado (`enviado`, `guardado`, `fallido`) queda en el log y en la tabla nueva `avisos`, enlazado a su métrica (`python -m app.cli avisos`); el CSV del panel tiene la columna "Aviso a la coordinadora". Un aviso que falla también se guarda como `.eml`. `SMTP_PASSWORD` nunca va al log, al `.eml` ni a la base (si un error del servidor la repitiera, se reemplaza por `***`). **Medido** con un SMTP simulado que tarda 3 s en contestar: `handle_message` pasa de 3.03 s a 0.005 s (el aviso agrega 0.000 s; antes, con un SMTP que no responde, el tope era 10 s por consulta); antes no quedaba ningún registro del aviso, después 1 fila por aviso. **En Docker** (código de la rama, base real): una consulta derivada generó su `.eml` en `data/outbox/` y su fila en `avisos` (`guardado`, `vista_previa`, `csistemas@tuxtla.tecnm.mx`), con la respuesta al estudiante en 0.02 s. Pruebas: `test_notify.py` (27, con un servidor SMTP local en un hilo, sin red: STARTTLS con certificado de prueba, login, destinatario según el modo, contenido, contraseña incorrecta, destinatario rechazado, SMTP caído, SMTP lento). **Falta**: poner la contraseña de aplicación de Gmail y `AVISOS_CORREO_PRUEBA`, correr `python -m app.cli probar-correo` y, ya revisado, `AVISOS_MODO=coordinadora` | HU-11 CA-01/CA-03, UC-12 |
| ✅ **Resuelto el 2026-10-24**. `CONTACT_EMAIL` vacío: la derivación no daba canal de contacto, solo horario. **Corrección**: `.env` (fuera de git) y `.env.example` con `CONTACT_EMAIL=csistemas@tuxtla.tecnm.mx` y `CONTACT_HOURS=lunes a viernes de 8:00 a 15:00 horas`; `app/rag/messages.py` ya los usaba. | RF-C-04, HU-08 CA-02, HU-11 CA-02 |
| ✅ **Resuelto el 2026-10-24**. Tiempo de búsqueda medido pero no registrado. **Corrección**: `RagPipeline.answer` lo escribe en el log (`INFO rag: búsqueda: 0.204 s, 5 fragmentos, similitud máx. 0.773`) y `record_metric` lo guarda en la columna nueva `metricas.tiempo_busqueda_seg` (SQLite y PostgreSQL; vacía si la consulta se derivó antes de buscar); el CSV tiene la columna "Tiempo de búsqueda (s)". **Medido** en el contenedor con el índice real (785 fragmentos), 62 preguntas de `data/retrieval_cases.jsonl` × 3: promedio 18.6 ms, p99 28.7 ms, máximo 39.1 ms, 0 de 186 sobre 2 s. | **RNF-EF-03** ("registrado en logs") |
| ✅ **Resuelto el 2026-10-24**. Respuestas de más de 4,096 caracteres sin manejo en Telegram: Telegram las rechazaba, el reintento en texto plano también excedía y el estudiante se quedaba sin respuesta. **Corrección**: `split_markdown` (`app/telegram_format.py`) parte el Markdown por párrafo, luego por línea y por palabra (sin separar un bloque de código), de modo que el HTML de cada trozo quepa en 4,000 caracteres (margen porque Telegram cuenta emojis como 2); cada trozo se convierte por separado, así que ninguna etiqueta queda abierta entre mensajes. `on_message` los envía en orden y los botones de calificación van solo en el último; si Telegram rechaza el HTML de un trozo, ese trozo se reenvía en texto plano. **Medido** con Telegram simulado que rechaza > 4,096 caracteres y una lista de 40 opciones (15,820 caracteres): antes, 0 mensajes entregados (2 rechazos y `BadRequest` sin atrapar); después, 4 mensajes (3,969-3,987 caracteres), las 40 opciones completas y los botones solo en el 4.º. Pruebas: `test_telegram_format.py` (5) y `test_telegram_bot.py::test_respuesta_larga_se_divide_y_los_botones_van_al_final`, `test_respuesta_larga_en_texto_plano_tambien_se_divide` | HU-06 CA-03, HU-02 |
| ✅ **Resuelto el 2026-10-24**. El tope de espera total (≤ 90 s) podía excederse: `FallbackLLM.chat` reiniciaba el presupuesto de 80 s en cada llamada y el pipeline hace hasta 2; además la fecha límite solo se revisaba antes de cada intento, y un intento podía durar `LLM_TIMEOUT` (25 s) completo después de ella. **Corrección**: `RagPipeline.answer` fija una sola fecha límite (`LLM_TOTAL_BUDGET`, 80 s) y la pasa a los 2 intentos; `FallbackLLM.chat(deadline=)` la respeta en todos los proveedores, y `OpenAICompatLLM.chat` recorta el timeout de cada petición y cada espera entre reintentos al tiempo restante. **Medido** con reloj simulado, 3 proveedores y `LLM_TIMEOUT`=25 s: si ninguno contesta, antes 104 s y después 80 s; si el 1.er intento llega a los 77 s con una respuesta inválida y en el 2.º nadie contesta, antes 181 s (el peor caso teórico rondaba 207 s) y después 80 s. Pruebas con reloj simulado: `test_llm.py` (4 nuevas; p. ej. 3 proveedores lentos terminan en ≤ 80 s en vez de > 3 min) y `test_pipeline.py::test_los_dos_intentos_comparten_un_solo_tope`. Nota: el timeout de httpx es por operación (conectar, leer), no por petición completa; con las respuestas no-streaming del LLM, en la práctica equivale | RNF-FI-02 |
| ✅ **Resuelto el 2026-10-24**. Bot de Telegram sin pruebas (indicador de escritura, calificación, errores). `test_telegram_bot.py` pasa de 3 a 13 pruebas, con updates simulados (`Update.de_json`) y un bot falso, sin red: el indicador "escribiendo…" se envía antes de la respuesta, se renueva mientras el pipeline trabaja y se detiene al responder o al fallar; botones 1-5 solo en respuestas (no en derivaciones); calificación registrada y botones retirados; calificación fallida o `callback_data` mal formado avisan sin tronar; error del servicio con mensaje al estudiante; HTML rechazado se reenvía en texto plano | HU-02 CA-03 |
| "Precisión" = % respondidas, sin verificar que sean correctas con consultas de la coordinadora | RF-RN-03 |
| ✅ **Resuelto el 2026-10-26** (rama `reportes-coordinadora`). Resumen del panel sin filtro de fechas (sí en CSV y API). **Corrección**: la sección "Reportes" del panel filtra por fechas en hora de Tuxtla y agrupa por semana, mes o ciclo (ver N10) | HU-09 CA-02, HU-13 CA-02 |

Dos tareas ya propuestas cubren varios de estos: "Registro de accesos, avisos por correo y
tiempo de búsqueda" y "Respuestas largas, tope de 90 s y pruebas del bot" (las dos, resueltas el 2026-10-24).

**Migraciones de la corrección de avisos, accesos y tiempo de búsqueda** (mismo patrón que N15): en
PostgreSQL la columna `tiempo_busqueda_seg` y las tablas `avisos` y `accesos_fallidos` se crean solo
si `information_schema` dice que faltan, dentro del candado consultivo y con `lock_timeout`; en
SQLite, en `SqliteRepository._migrate`. En Docker, la base real pasó de 6 a 8 tablas y conservó sus
16 métricas, 32 mensajes, 1 usuario y 3 conversaciones. `test_repository_postgres.py` (11, en Docker
contra una base de prueba aparte): migración sin perder datos y arranque sin esperar con lecturas
abiertas sobre las tablas nuevas.

**Formato de los avisos y resumen diario** (2026-10-26, HU-11 CA-01/CA-03, UC-12): asunto
"[Chatbot ISC] <Tipo>: <pregunta recortada a 60 caracteres, sin saltos de línea>", remitente
"Chatbot ISC – ITTG", cuerpo HTML (estilos en línea, todo el texto del estudiante y del chatbot
escapado) con versión de texto plano. Tres tipos según el motivo de derivación: datos personales o
trámite oficial (`out_of_scope`), sin información en los documentos (`low_similarity`,
`not_in_context`, `validation_failed`) y falla técnica (`llm_unavailable`). `AVISOS_INMEDIATOS`
(por defecto `falla_tecnica`) se avisan al momento; el resto va al resumen diario de
`AVISOS_RESUMEN_HORA` (por defecto 08:00, con el día anterior de 00:00 a 24:00 en hora local), que
se registra en `avisos` (`tipo_aviso = resumen`, `fecha_resumen`) para no mandarlo dos veces si el
bot se reinicia; un día sin derivaciones no manda nada. Al estudiante se le agrega "Le enviaremos un
aviso con tu consulta a la Coordinación de ISC…" (o "se incluirá en el resumen diario…") solo en
modo `coordinadora` con el correo configurado. Columnas nuevas `avisos.tipo_aviso` y
`avisos.fecha_resumen`, solo si faltan; en Docker la base real las recibió sin perder sus filas.
Ejemplos: `python -m app.cli ejemplos-avisos` (`data/outbox/ejemplo_*.eml`, sin tocar la base).
Pruebas: `test_notify.py`, `test_repository.py` y `test_repository_postgres.py`.

**Límite de avisos de falla técnica** (2026-10-26): como máximo un aviso inmediato de falla técnica
cada `AVISOS_FALLA_INTERVALO_MIN` minutos (30 por defecto; 0 = sin límite). El siguiente aviso dice
cuántas consultas se afectaron desde el último ("7 consultas afectadas desde el último aviso de falla
técnica (…), contando esta"). El estado sale de la base (hora del último aviso enviado o guardado en
`avisos` y conteo de `metricas` con motivo `llm_unavailable`), así que se respeta aunque el bot se
reinicie; un envío fallido no cuenta como aviso. Pruebas con reloj simulado en `test_notify.py`.

**Mensajes de orientación por instancia** (2026-10-26, RF-C-04, HU-08): datos personales y
documentos oficiales remiten a "Servicios Escolares (Control Escolar) del ITTG" sin correo ni el
horario de la coordinación; las dudas sin respuesta en los documentos y la falla técnica remiten a
la Coordinación de ISC con `CONTACT_HOURS` y `CONTACT_EMAIL`; inyección y mensaje demasiado largo no
dan contacto. Pruebas: `test_messages.py` (13).

✅ **Resuelto el 2026-10-26: candado del esquema al arrancar** (ya pasaba con `main`): el
`CREATE INDEX IF NOT EXISTS` del esquema original pedía un candado SHARE sobre la tabla aunque el
índice existiera, así que el arranque esperaba a cualquier escritura en curso del otro servicio.
**Corrección** (`app/adapters/db_postgres.py`): cada tabla e índice del esquema se crea solo si
`to_regclass` dice que falta (consultarlo no pide candados). **Medido** en Docker contra una base
de prueba aparte, con `test_repository_postgres.py::test_arranque_no_espera_a_una_escritura_abierta`
(INSERT sin COMMIT en `conversaciones`, `mensajes` y `metricas`): antes de la corrección el arranque
esperó los 2.0 s del `lock_timeout` de la prueba y falló con `LockNotAvailable`; con la corrección
arranca sin esperar y pasan las 13 pruebas de PostgreSQL.

## 2. Numeración de HU: código (Lista A) → protocolo (Lista B)

| Cita en el código | Dónde | Corresponde en la Lista B |
|---|---|---|
| HU-01 CA-03 (indicador de escritura) | `app/telegram_bot.py` | HU-02 CA-03 |
| HU-01 CA-04 (mensaje demasiado largo) | `app/domain.py`, `app/service.py` | HU-02 CA-04 |
| HU-04 (métricas) | repositorios, `app/admin_service.py`, `app/web_admin.py`, `tests/test_metrics_export.py` | HU-09 / HU-13 |
| HU-07 (en "métricas (RF-I-04, HU-04, HU-07…)") | repositorios | HU-09 CA-01 (registro); HU-07 es la validación |
| HU-12 en la Lista A (actualizar KB) | — | HU-14 |
| HU-05, 08, 10, 11, 13, 14, 15 | varios | Coinciden |

## 3. Desviaciones del protocolo (decisiones de implementación)

| Protocolo | Implementado | Justificación / evidencia |
|---|---|---|
| Similitud mínima 0.75 | 0.80 | Calibrado: la pregunta válida más baja midió 0.840 y lo claramente ajeno 0.77-0.80 (`data/retrieval_cases.jsonl`) |
| text-embedding-3-small (API) | `intfloat/multilingual-e5-small` local | Sin costo ni dependencia de una API que pueda retirarse (`README.md`) |
| GPT-4o mini, temp. 0.3 | Groq `openai/gpt-oss-120b`, temp. 0.3, con respaldo OpenAI y Ollama | `docs/capacidad-y-costos.md` |
| LlamaIndex | Pipeline propio (`app/rag/`) | — |
| RDS + EC2 + S3 | PostgreSQL 16 en Docker local | Despliegue en AWS pendiente (N4-N6) |
| Telegram → API REST FastAPI | Telegram en proceso con `ConversationService` | Pendiente (N3) |

## 4. Requisitos de información (RF-I)

| Código | Requisito | Estado | Implementación | Pruebas / evidencia | Falta |
|---|---|---|---|---|---|
| RF-I-01 | Consultas en lenguaje natural por Telegram, sin comandos | ✅ | `app/telegram_bot.py` (`MessageHandler(filters.TEXT & ~filters.COMMAND)`, `update.effective_message`), `app/service.py` | `test_service.py::test_handle_message_persists_and_returns_metric_id` (motor); `test_telegram_bot.py` (adaptador: mensaje nuevo y editado); 10 consultas reales en `metricas` | — (no hay cola: se atiende en paralelo con `asyncio.to_thread`) |
| RF-I-02 | Historial de la conversación activa para el LLM | ✅ | `get_history` (repositorios), `RagPipeline._build_messages` | `test_repository.py::test_history_order_and_trim`, `test_pipeline.py::test_history_is_included_and_trimmed` | — (la búsqueda usa el historial solo en preguntas de seguimiento: ver HU-04 CA-02) |
| RF-I-03 | Búsqueda de similitud, 5 fragmentos, similitud mínima | ✅ | `RagPipeline.retrieve` y filtro por `SIM_THRESHOLD` (`app/rag/pipeline.py`), `TOP_K=5` | `test_pipeline.py::test_low_similarity_derives_without_calling_llm`, `test_retrieval_real.py` (54/54 en el top 5) | Umbral 0.80 en vez de 0.75 (sección 3) |
| RF-I-04 | Registrar mensajes y respuestas con tiempo, fragmentos, similitud máxima y calificación | ✅ | `add_message`, `record_metric`, `rate_response` (repositorios); tablas `mensajes`, `metricas`, `metricas_fragmentos` | `test_repository.py::test_record_metric_answered_stores_fragments_and_provider`, `test_rate_response_bounds`, `test_service.py::test_rate_updates_metric` | — |
| RF-I-05 | Cargar PDF/DOCX, fragmentar, vectorizar e indexar sin código ni reinicio; error con el nombre del archivo | ✅ | `app/admin_service.py`, `app/web_admin.py`, `app/kb_watcher.py` | `test_admin_service.py` (14), `test_web_admin.py::test_upload_*`, `test_kb_watcher.py` (5) | — |

## 5. Requisitos de comportamiento (RF-C)

| Código | Requisito | Estado | Implementación | Pruebas / evidencia | Falta |
|---|---|---|---|---|---|
| RF-C-01 | Identificar intención y entidades (inscripción, baja, residencia, servicio social, titulación) | 🟡 | Expansión por tema y `classify_topic` (9 temas, guardado en `metricas.tema`) en `app/rag/query_expansion.py`; reglas de alcance en `app/rag/guards.py` | `test_query_expansion.py` (8), `test_topics.py` (27; 62/62 en `retrieval_cases.jsonl`) | Sin extracción de entidades; inscripción va dentro de "plan de estudios" (N11). El protocolo acepta "sin módulo separado" |
| RF-C-02 | Respuesta del LLM con consulta, historial y fragmentos; temperatura 0.3 | ✅ | `RagPipeline.answer`, `SYSTEM_PROMPT`; `LLM_TEMPERATURE=0.3` | `test_pipeline.py::test_answered_with_sources_and_prompt_shape`, `test_history_is_included_and_trimmed` | — |
| RF-C-03 | Validar (fundamentada, tono institucional) y regenerar | ✅ | `RagPipeline.validate` | `test_pipeline.py::test_invented_figure_fails_validation_then_regenerates`, `test_ungrounded_answer_fails_validation`, `test_lista_recortada_*`, volcado del contexto (B8) | El tono solo se pide en las instrucciones; no se valida |
| RF-C-04 | Derivar por similitud baja o consulta excepcional; mensaje con instancia y canal | ✅ | `app/rag/messages.py`, `app/rag/guards.py`; `CONTACT_EMAIL=csistemas@tuxtla.tecnm.mx` en `.env` y `.env.example` | `test_pipeline.py::test_low_similarity_derives_without_calling_llm`, `test_out_of_scope_rules_skip_retrieval_and_llm` | — |
| RF-C-05 | Disponibilidad 24/7, respuesta ≤ 30 s | ❌ | `restart: unless-stopped` en `docker-compose.yml` | ≤ 30 s: sí (RNF-EF-01) | Despliegue permanente (N4) |
| RF-C-06 | Sesiones con id único, inicio/fin y mensajes asociados | ✅ | `get_or_create_conversation` (cierra con `fecha_fin` por inactividad), tabla `conversaciones` | `test_repository.py::test_conversation_reused_within_timeout_new_session_after`, `test_close_conversation_forces_new_session` | — |

## 6. Reglas del negocio (RF-RN)

| Código | Regla | Estado | Implementación | Pruebas / evidencia | Falta |
|---|---|---|---|---|---|
| RF-RN-01 | No dar calificaciones, kárdex, adeudos ni datos del SIIA; orientar | ✅ | `app/rag/guards.py` (`datos_personales`), `app/rag/messages.py` | `test_pipeline.py::test_out_of_scope_rules_skip_retrieval_and_llm` | — |
| RF-RN-02 | Respuestas solo con documentación oficial vigente | ✅ | Regla 1 del prompt, `validate` (sustento y cifras), reemplazo de documentos por versión | `test_pipeline.py::test_ungrounded_answer_fails_validation`, `test_store_kb.py::test_ingest_added_unchanged_replaced_and_persisted`; red-team B6 (forzar alucinación) | — |
| RF-RN-03 | Precisión ≥ 90 % = respondidas correctamente sin derivación, con consultas de la coordinadora | 🟡 | `precision_rate` (`app/adapters/db_sqlite.py`), `python -m app.cli report` | `test_repository.py::test_metrics_summary_rates_and_reasons`; recuperación 54/54 | Conjunto de consultas de la coordinadora y verificar que las respuestas sean **correctas** (hoy solo cuenta respondidas) |

## 7. Requisitos no funcionales

| Código | Requisito / criterio | Estado | Implementación | Pruebas / evidencia | Falta |
|---|---|---|---|---|---|
| RNF-FI-01 | Disponibilidad 99 % mensual, monitoreada | ❌ | Reiniciar un servicio ya no lo deja colgado esperando al otro: lecturas sin transacción abierta y `lock_timeout` al crear el esquema (N15) | `test_repository_postgres.py::test_arranque_no_espera_a_la_lectura_de_otro_servicio`, `::test_bloqueo_al_arrancar_falla_con_error_claro` | Despliegue en AWS y monitoreo (N4) |
| RNF-FI-02 | 3 reintentos con espera exponencial; espera total ≤ 90 s; aviso al usuario | ✅ | `app/adapters/llm_openai_compat.py` (`LLM_MAX_RETRIES=3`; fecha límite compartida que recorta timeouts y esperas), `app/rag/pipeline.py` (un solo tope de `LLM_TOTAL_BUDGET`=80 s para los 2 intentos), mensaje de servicio no disponible | `test_llm.py` (11, 4 con reloj simulado), `test_pipeline.py::test_los_dos_intentos_comparten_un_solo_tope`, `test_llm_down_derives_gracefully` | — |
| RNF-FI-03 | Integridad referencial y respaldo diario | 🟡 | Llaves foráneas con `ON DELETE CASCADE`; migración de columnas solo si faltan, dentro del candado del esquema (N15) | `test_repository.py::test_foreign_key_cascade_deletes_children`, `test_repository_postgres.py::test_migracion_agrega_la_columna_si_falta` | Respaldos (N5) |
| RNF-US-01 | 90 % de usuarios nuevos logra una consulta útil sin instrucciones | 🟡 | Texto libre, sin comandos | — | Prueba con usuarios (N12) |
| RNF-US-02 | Claridad y tono; 85 % califica ≥ 4 | 🟡 | Calificación 1-5 guardada en `metricas` | — | Reporte del % ≥ 4 y encuesta (N12) |
| RNF-US-03 | Funciona en Android, iOS, web y escritorio | 🟡 | Telegram como canal | — | Prueba de integración en los 4 (N12) |
| RNF-US-04 | Coordinadora actualiza la KB sin conocimientos técnicos; 50 págs en ≤ 10 min | 🟡 | Panel (`app/web_admin.py`) con confirmación de éxito/error | `test_web_admin.py` (12) | Medir el tiempo de carga de un documento de 50 páginas (N12) |
| RNF-SE-01 | Autenticación por ID de Telegram; intentos no autorizados en logs | 🟡 | Registro automático en `usuarios`; intentos fallidos del panel en el log y en `accesos_fallidos` | `test_repository.py::test_user_creation_is_idempotent_by_telegram_id`, `test_web_admin.py::test_intento_fallido_queda_registrado_sin_la_contrasena` | Definir qué es "usuario no registrado" en Telegram con registro automático |
| RNF-SE-02 | No guardar datos sensibles; cifrado en reposo | 🟡 | No se piden datos sensibles; las preguntas de datos personales se derivan | — | Cifrado en reposo (N6) |
| RNF-SE-03 | HTTPS/TLS; credenciales solo en variables de entorno | 🟡 | `.env` fuera de git; sin credenciales en el historial (revisado); HTTPS hacia Telegram, Groq y SMTP (STARTTLS) | `.gitignore`, `.dockerignore` | HTTPS en el panel (N7) |
| RNF-SE-04 | 20 casos de inyección neutralizados | ✅ | `app/rag/guards.py`, `RagPipeline.validate`, `app/redteam.py` | `data/redteam_2026-10-22_2346.json` (20/20, LLM real); `test_redteam.py` (16) | — |
| RNF-EF-01 | 95 % de consultas ≤ 30 s, medido en METRICAS | ✅ | `metricas.tiempo_respuesta_seg`; p95 en el panel, `/api/metricas` y el CSV del reporte (`app/reports.py`) | p95 = 2.00 s en las 17 consultas reales (calculado por el sistema, N13); `data/loadtest_2026-10-22_2332.json` (p95 1.78 s); `test_reports.py::test_resumen_trae_p95` | — |
| RNF-EF-02 | 20 concurrentes, promedio ≤ 30 s, sin errores 5xx | ✅ | — | `data/loadtest_2026-10-22_2332.json` (0/20 errores, LLM simulado); `docs/capacidad-y-costos.md` (LLM real: 6.96 s, 0 errores) | Con Groq gratis la mayoría se deriva por el límite (429) |
| RNF-EF-03 | Búsqueda vectorial ≤ 2 s en el 99 %, índice en RAM, registrado en logs | ✅ | FAISS `IndexFlatIP` en memoria; tiempo en el log (`rag: búsqueda: ...`) y en `metricas.tiempo_busqueda_seg` | `test_repository.py::test_record_metric_guarda_el_tiempo_de_busqueda`; medido en el contenedor: p99 28.7 ms (186 búsquedas) | — |
| RNF-PO-01 | Despliegue en EC2 solo con Dockerfile y docker-compose.yml | 🟡 | `Dockerfile`, `docker-compose.yml` (db, bot, admin) | Arranque verificado en Docker local | EC2 (N4) |
| RNF-PO-02 | Otra carrera = reindexar y cambiar variables | 🟡 | Documentos y umbrales por configuración | — | Textos de ISC fijos (N8) |
| RNF-PO-03 | Telegram por API REST de FastAPI; interfaz sin lógica ni acceso a BD/IA | 🟡 | `ConversationService` común (`app/service.py`), adaptador delgado | `test_service.py` (5) | API REST entre canal y backend (N3) |

## 8. Historias de usuario (Lista B) y criterios de aceptación

| HU | CA | Criterio | Estado | Implementación / prueba | Falta |
|---|---|---|---|---|---|
| HU-01 Autenticarme por Telegram | CA-01 | Registro automático por ID | ✅ | `get_or_create_user`; `test_repository.py::test_user_creation_is_idempotent_by_telegram_id` | — |
| | CA-02 | Sin contraseñas | ✅ | Solo ID de Telegram | — |
| | CA-03 | Accesos no autorizados en logs | 🟡 | Panel: log y tabla `accesos_fallidos` (`app/web_admin.py`); `test_web_admin.py::test_intento_fallido_queda_registrado_sin_la_contrasena` | Telegram: definir "no autorizado" con registro automático (RNF-SE-01) |
| | CA-04 | No se vuelve a registrar | ✅ | Misma prueba que CA-01 | — |
| HU-02 Enviar consultas | CA-01 | Texto libre | ✅ | `app/telegram_bot.py`; `test_telegram_bot.py::test_mensaje_nuevo_o_editado_recibe_respuesta` | — (N2 resuelto) |
| | CA-02 | < 30 s | ✅ | RNF-EF-01 | — |
| | CA-03 | Indicador de escritura | ✅ | `_typing_loop` (`app/telegram_bot.py`); `test_telegram_bot.py::test_escribiendo_se_envia_mientras_piensa_y_se_detiene_al_responder`, `test_escribiendo_se_detiene_si_el_servicio_falla` | — |
| | CA-04 | Límite de caracteres: pedir dividir | ✅ | `app/service.py`; `test_service.py::test_too_long_message_is_derived_without_calling_llm` | — |
| HU-03 Requisitos de trámites | CA-01 | Fundamentada en documentos oficiales | ✅ | `validate`; RF-RN-02 | — |
| | CA-02 | Pasos y documentos del trámite | 🟡 | Reglas 3 y 4 del prompt (pasos numerados, listas completas); `test_pipeline.py::test_prompt_pide_listas_completas` | Evaluar respuestas reales de trámites |
| | CA-03 | Datos personales se derivan | ✅ | `test_pipeline.py::test_out_of_scope_rules_skip_retrieval_and_llm` | — |
| | CA-04 | < 30 s | ✅ | RNF-EF-01 | — |
| HU-04 Preguntas de seguimiento | CA-01 | Historial desde la BD | ✅ | `test_repository.py::test_history_order_and_trim` | — |
| | CA-02 | Respuestas coherentes | ✅ | `search_query` (`app/rag/query_expansion.py`); `test_pipeline.py::test_seguimiento_busca_con_la_pregunta_anterior`, `test_pregunta_independiente_no_mezcla_el_historial`, `test_seguimiento_encadenado_conserva_el_tema`; en el contenedor 7/7 seguimientos con el fragmento correcto y respuestas correctas del LLM real a "¿Y cuántas horas son?" y "¿Y si lo repruebo?" | — (N1 resuelto) |
| | CA-03 | Sesión expirada abre una nueva | ✅ | `test_repository.py::test_conversation_reused_within_timeout_new_session_after` | — |
| | CA-04 | Historial en el contexto del LLM | ✅ | `test_pipeline.py::test_history_is_included_and_trimmed` | — |
| HU-05 Intención y entidades | CA-01 | Clasifica los 5 tipos | 🟡 | `classify_topic` (`app/rag/query_expansion.py`): baja, residencia, servicio social y titulación con tema propio; inscripción dentro de "plan de estudios"; `test_topics.py` | N11 |
| | CA-02 | Entidades disponibles para RAG | 🟡 | La expansión aporta vocabulario del tema | Entidades explícitas |
| | CA-03 | Fuera de dominio deriva | ✅ | Umbral, `guards.py`, NO_INFO; `test_pipeline.py::test_low_similarity_derives_without_calling_llm` | — |
| | CA-04 | PLN añade ≤ 5 s | ✅ | Reglas locales (milisegundos) | — |
| HU-06 Respuestas contextualizadas | CA-01 | Solo fragmentos con similitud mínima | ✅ | Filtro por umbral en `RagPipeline.answer` | — |
| | CA-02 | Tono formal | 🟡 | Regla 3 del prompt | Evaluación |
| | CA-03 | Extensión para móvil | 🟡 | ≤ 150 palabras (salvo listas completas); las respuestas de más de 4,096 caracteres se dividen en varios mensajes (`split_markdown`, `test_telegram_bot.py::test_respuesta_larga_se_divide_y_los_botones_van_al_final`) | Evaluar la lectura en el celular |
| | CA-04 | Temperatura 0.3 | ✅ | `LLM_TEMPERATURE=0.3` | — |
| HU-07 Validar antes de enviar | CA-01 | Rechazar sin fundamento | ✅ | `test_pipeline.py::test_ungrounded_answer_fails_validation` | — |
| | CA-02 | Regenerar | ✅ | `test_pipeline.py::test_invented_figure_fails_validation_then_regenerates` | — |
| | CA-03 | Si falla el 2.º intento, derivar | ✅ | `test_pipeline.py::test_two_failed_validations_derive` | — |
| | CA-04 | Resultado registrado | ✅ | `metricas.resultado`, `intentos`, motivo `validation_failed` | — |
| HU-08 Orientación sin respuesta | CA-01 | Indica instancia | ✅ | `app/rag/messages.py` (Coordinación / Servicios Escolares) | — |
| | CA-02 | Canal y horario | ✅ | `app/rag/messages.py` con `CONTACT_EMAIL` y `CONTACT_HOURS` configurados | — |
| | CA-03 | Deriva por similitud baja o fuera de dominio | ✅ | `test_pipeline.py::test_low_similarity_derives_without_calling_llm`, `test_out_of_scope_rules_skip_retrieval_and_llm` | — |
| | CA-04 | Registro con indicador de derivación | ✅ | `test_repository.py::test_record_metric_derived_stores_reason_and_no_provider` | — |
| HU-09 Registrar y consultar métricas | CA-01 | Registro automático | ✅ | `test_service.py::test_handle_message_persists_and_returns_metric_id` | — |
| | CA-02 | Reportes filtrables por periodo | ✅ | Sección "Reportes" del panel, CSV y `/api/metricas`; `test_reports.py::test_filtro_por_fechas_en_hora_local`, `test_panel_con_fechas_vacias_y_periodo_sin_datos`, `test_metrics_export.py::test_filtro_por_dia_usa_la_hora_de_tuxtla` | — |
| | CA-03 | Tasa de derivación visible | ✅ | Tarjeta en `app/templates/panel.html`; `test_web_admin.py::test_panel_loads_with_correct_credentials` | — |
| | CA-04 | Último reporte si no hay datos | ✅ | `build_report` (`app/reports.py`): "Último reporte disponible" en el panel; `test_reports.py::test_periodo_vacio_muestra_el_ultimo_reporte_disponible` | — (N10) |
| HU-10 Gestionar sesiones | CA-01 | Sesión nueva al primer mensaje | ✅ | `test_repository.py::test_conversation_without_prior_messages_reused` | — |
| | CA-02 | Mensajes con el mismo id | ✅ | `test_service.py::test_handle_message_reuses_user_and_session_across_calls` | — |
| | CA-03 | Cierre por inactividad configurable | ✅ | `SESSION_TIMEOUT_MINUTES` | — |
| | CA-04 | Reanudar crea sesión nueva | ✅ | `test_repository.py::test_conversation_reused_within_timeout_new_session_after` | — |
| HU-11 Escalamientos con contexto | CA-01 | Contexto por correo institucional | 🟡 | `app/notify.py` (Gmail; modos vista previa, prueba y coordinadora; en segundo plano); `test_notify.py::test_prueba_envia_al_correo_de_prueba_con_starttls_y_login`, `test_smtp_lento_no_retrasa_la_respuesta_y_el_aviso_queda_registrado` | Prueba de entrega real: `python -m app.cli probar-correo` con la contraseña de aplicación |
| | CA-02 | Horario y correo al estudiante | ✅ | `app/rag/messages.py`; `CONTACT_EMAIL` configurado | — |
| | CA-03 | Registro en METRICAS | ✅ | Tabla `avisos` enlazada a `metricas` y columna "Aviso a la coordinadora" en el CSV; `test_repository.py::test_avisos_enlazados_a_su_metrica`, `test_notify.py::test_vista_previa_desde_el_servicio_guarda_eml_y_registro` | — |
| | CA-04 | No escalar lo resoluble | ✅ | `should_notify`; `test_notify.py::test_should_not_notify_for_non_escalable_reasons` | — |
| HU-12 Cualquier dispositivo | CA-01 | Android, iOS, web, escritorio | 🟡 | Telegram | Prueba de integración (N12) |
| | CA-02 | Sin instalar nada | ✅ | Telegram Web | — |
| | CA-03 | 24/7 | ❌ | — | N4 |
| | CA-04 | < 30 s | ✅ | RNF-EF-01 | — |
| HU-13 Reportes de desempeño | CA-01 | Total, precisión, tiempo, derivación y tipos más frecuentes | ✅ | Panel (tarjetas y sección "Reportes" con tipos de consulta y p95) y CSV; `test_reports.py::test_tipos_mas_frecuentes_y_temas_con_mas_derivaciones`, `test_panel_muestra_los_reportes` | — (N9) |
| | CA-02 | Filtro por fechas | ✅ | Panel (hora de Tuxtla, campos vacíos = sin fecha), CSV y API; `test_reports.py::test_filtro_por_fechas_en_hora_local` | — |
| | CA-03 | Precisión = % resueltas sin derivación | ✅ | `precision_rate`; `test_repository.py::test_metrics_summary_rates_and_reasons` | (Ver RF-RN-03: "correctamente") |
| | CA-04 | Exportable | ✅ | `/metricas/exportar.csv` (con "Tema") y `/reportes/exportar.csv` (por periodo, con promedio y p95); `test_metrics_export.py` (15), `test_reports.py::test_csv_del_reporte`, `test_descarga_del_reporte` | — |
| HU-14 Actualizar la base de conocimiento | CA-01 | PDF y DOCX | ✅ | `app/rag/loaders.py`; `test_store_kb.py::test_ingest_docx` | — |
| | CA-02 | Fragmenta y vectoriza solo | ✅ | `test_admin_service.py::test_upload_new_document_indexes_and_syncs_to_db` | — |
| | CA-03 | Índice se recarga de inmediato | ✅ | `app/kb_watcher.py`; `test_kb_watcher.py` (5) | — |
| | CA-04 | Error con nombre de archivo | ✅ | `test_web_admin.py::test_upload_unsupported_extension_shows_error_and_is_not_indexed` | — |
| | CA-05 | Sin tocar el código | ✅ | Panel web | — |
| HU-15 Desempeño histórico | CA-01 | Métricas desde el despliegue | ✅ | Todas quedan en `metricas`; CSV sin fechas = todo | — |
| | CA-02 | Agrupadas por semana, mes o ciclo | ✅ | `period_of` y `SCHOOL_CYCLES` (`app/reports.py`); `test_reports.py::test_periodos`, `test_cortes_en_hora_de_tuxtla` | — (N10) |
| | CA-03 | Tipos más frecuentes y áreas con más derivación | ✅ | Temas en `metricas.tema`; tablas del panel y del CSV del reporte; `test_reports.py::test_tipos_mas_frecuentes_y_temas_con_mas_derivaciones` | — (N9) |
| | CA-04 | Indicar periodo mínimo si faltan datos | ✅ | `REPORTES_MIN_CONSULTAS` (20); aviso con el periodo mínimo necesario; `test_reports.py::test_periodo_insuficiente_dice_el_periodo_minimo`, `test_ningun_periodo_alcanza_el_minimo` | — (N10) |

## 9. Casos de uso

| UC | Nombre | Estado | Implementación |
|---|---|---|---|
| UC-01 | Autenticar estudiante | 🟡 | `get_or_create_user`; intentos fallidos del panel registrados (en Telegram falta definir "no autorizado") |
| UC-02 | Enviar consulta | ✅ | `app/telegram_bot.py` (mensajes nuevos y editados; `test_telegram_bot.py`) |
| UC-03 | Consultar trámite | ✅ | Pipeline RAG |
| UC-04 | Conversación contextual | ✅ | Historial al LLM y, en preguntas de seguimiento, a la búsqueda (`search_query`) |
| UC-05 | Orientar a área responsable | ✅ | `app/rag/messages.py` con `CONTACT_EMAIL` y horario |
| UC-06 | Notificar respuesta | ✅ | Respuesta en Telegram con HTML y calificación |
| UC-07 | Procesar con PLN | 🟡 | Reglas por tema: expansión de la búsqueda y `classify_topic` (9 temas); sin entidades (N11) |
| UC-08 | Buscar en FAISS | ✅ | `app/adapters/vectorstore_faiss.py`, `RagPipeline.retrieve` |
| UC-09 | Generar respuesta (LLM) | ✅ | `RagPipeline.answer`, `FallbackLLM` |
| UC-10 | Registrar interacciones en PostgreSQL | ✅ | `app/adapters/db_postgres.py` (pruebas en Docker 20/20) |
| UC-11 | Gestionar sesión | ✅ | `get_or_create_conversation` |
| UC-12 | Escalar a coordinadora | 🟡 | `app/notify.py` (Gmail, en segundo plano, registro en `avisos`); falta la prueba de entrega real |
| UC-13 | Validar respuesta | ✅ | `RagPipeline.validate` |
| UC-14 | Consultar métricas | ✅ | Panel (resumen y "Reportes" por semana, mes o ciclo, con tipos de consulta, temas con más derivaciones y p95) y dos CSV |
| UC-15 | Actualizar base FAISS | ✅ | Panel + `KBWatcher` |

## 10. Evidencia en el repositorio

| Archivo | Qué demuestra |
|---|---|
| `data/redteam_2026-10-22_2346.json` | RNF-SE-04: 20/20 con LLM real, 0 inconclusos |
| `data/redteam_2026-10-22_2328.json` | Corrida previa (19/20, falla B8) |
| `data/loadtest_2026-10-22_2332.json` | RNF-EF-01/02: 20 usuarios, base completa, p95 1.78 s, 0 errores (LLM simulado) |
| `data/retrieval_cases.jsonl` + `tests/test_retrieval_real.py` | RF-I-03: 54/54 preguntas con el fragmento correcto en el top 5; con `tests/test_topics.py`, 62/62 temas bien clasificados (N11) |
| `docs/capacidad-y-costos.md` | RNF-EF-02 con LLM real y límites del plan gratuito de Groq |
| Tabla `metricas` (base real) | RNF-EF-01: 17 consultas, prom. 0.99 s, p95 2.00 s (calculado por el panel, 2026-10-26) |
| Suite de pruebas | 415 aprobadas, 15 omitidas sin PostgreSQL (`test_repository_postgres.py` 15/15 en Docker con PostgreSQL) |
