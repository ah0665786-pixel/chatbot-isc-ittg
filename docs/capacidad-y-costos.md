# Capacidad y costos del proveedor LLM (Groq)

## Hallazgo (30 de septiembre de 2026)

Al correr las 20 pruebas de inyección de prompts (`python -m app.cli redteam`) contra
el plan gratuito de Groq, el sistema recibió varios `429 Too Many Requests` después de
apenas 5 llamadas seguidas. La cadena de respaldo configurada (`FallbackLLM`) absorbió el
problema con reintentos y espera progresiva, y las 20 pruebas terminaron pasando — pero el
hallazgo expone un límite real que hay que resolver antes de un despliegue con usuarios de
verdad.

## Causa raíz

Fuente: documentación oficial de Groq, `console.groq.com/docs/rate-limits` y
`console.groq.com/docs/models` (consultado 30 de septiembre de 2026).

Límites del modelo `openai/gpt-oss-120b` por plan:

| Límite | Plan gratis | Plan Developer (pago por uso) |
|---|---:|---:|
| Peticiones / minuto | 30 | 1,000 |
| Tokens / minuto | 8,000 | 250,000 |
| Peticiones / día | 1,000 | sin techo relevante |
| Tokens / día | 200,000 | sin techo relevante |
| Costo | $0 | $0.15 / $0.60 USD por millón de tokens (entrada/salida); sin cuota fija |

Cada consulta de este sistema consume aproximadamente 2,000–2,500 tokens (contexto de 5
fragmentos + historial de conversación + pregunta + respuesta generada). El límite que se
alcanzó fue el de **tokens por minuto**, no el de peticiones: 8,000 ÷ ~2,500 ≈ 3 consultas
por minuto en el plan gratis, antes de empezar a recibir `429`.

## Capacidad resultante por plan

- **Plan gratis**: ~3 consultas/minuto sostenidas. Suficiente para desarrollo y pruebas
  individuales, insuficiente para `RNF-EF-02` (20 consultas concurrentes).
- **Plan Developer**: ~100 consultas/minuto sostenidas (250,000 ÷ 2,500). Cubre con margen
  los 20 concurrentes de `RNF-EF-02`, incluyendo picos durante periodos de inscripción.

## Población total vs. carga concurrente pico

Es un error común (y lo fue en nuestra primera lectura del problema) asumir que "más
estudiantes matriculados" significa "más carga por segundo" de forma directa. Lo que
determina la capacidad necesaria es cuántas personas preguntan **al mismo tiempo**, no
cuántas están inscritas. La carga pico concurrente crece mucho más lento que la población
total, porque cada estudiante solo genera tráfico mientras escribe activamente.

Con el plan Developer, ~100 consultas/minuto sostenidas dejan margen de sobra incluso si la
población real creciera varias veces por encima del escenario de diseño (752 estudiantes de
ISC, o los 5,000 del ITTG completo si se cumple el escalamiento de `RNF-PO-02`).

## Costo estimado real

A $0.0006 USD por consulta promedio:

- 752 estudiantes de ISC, 2 consultas/semana, semestre de 16 semanas ≈ 24,000 consultas
  ≈ **menos de $15 USD por semestre**.
- Escalado a los ~5,000 estudiantes de todo el ITTG (RNF-PO-02), con el mismo supuesto de
  uso ≈ **menos de $100 USD por semestre**.

No es una suscripción: se paga únicamente por los tokens que de verdad se consumen.

## Decisión (30 de septiembre de 2026 — superada parcialmente, ver actualización abajo)

1. **Desarrollo y pruebas** (lo que hemos hecho hasta ahora): plan gratuito, consciente de
   su límite de 3 consultas/minuto.
2. ~~Antes de cualquier prueba de carga real (RNF-EF-02) o despliegue con estudiantes
   reales: activar el plan Developer en la cuenta de Groq
   (`console.groq.com/settings/billing/plans`), agregando una tarjeta.~~ **Ver hallazgo del
   3 de octubre: esta ruta está bloqueada por Groq temporalmente, no es un tema de
   presupuesto.**
3. **Configurar un límite de gasto** (`Spend Limits`, en la misma sección de la consola de
   Groq) como red de seguridad, para no arriesgar un cobro inesperado en una cuenta sin
   presupuesto institucional asignado todavía. (Queda pendiente hasta que se pueda activar
   el plan Developer.)
4. La cadena de respaldo de proveedores (`LLM_1/2/3` en `.env`) se mantiene de todas formas
   como segunda capa de defensa: ningún proveedor, ni siquiera de pago, está exento de
   caídas o incidentes — el plan Developer resuelve el límite de capacidad, no reemplaza la
   necesidad de tener un respaldo.

## Hallazgo (3 de octubre de 2026)

### 1. Groq pausó temporalmente las altas al plan Developer

La consola de Groq (`console.groq.com/settings/billing/plans`) muestra el plan Developer
("Pago por ficha") como no disponible para nuevas altas por alta demanda del proveedor:

> "Las actualizaciones al nivel de desarrollador no están disponibles temporalmente debido
> a la alta demanda. Pedimos disculpas por las molestias."

Esto **no es un problema de presupuesto ni de la cuenta institucional**: aunque se agregue
una tarjeta válida, Groq no permite la alta en este momento. Es una restricción del lado del
proveedor, históricamente temporal, pero sin fecha de reapertura anunciada.

### 2. Prueba de concurrencia real (20 solicitudes simultáneas) confirma el límite

Resultados de la prueba de carga con 20 solicitudes concurrentes:

- **Tiempo de respuesta**: promedio 6.96 s, máximo 8.83 s — cumple `RNF-EF-01` (< 30 s) con
  margen amplio.
- **Errores visibles al estudiante**: 0/20 — cumple `RNF-EF-02` en cuanto a estabilidad; el
  sistema nunca se cayó ni devolvió un error crudo.
- **Pero**: bajo esa carga, Groq (plan gratis) se saturó casi de inmediato (`429`), el
  respaldo intentó OpenAI y falló por clave inválida en `.env` (`401`), y el respaldo a
  Ollama local falló porque el modelo configurado no está instalado (`404`). Al fallar las
  tres capas, el sistema derivó correctamente a la coordinadora con el mensaje de
  orientación — comportamiento correcto (degradación elegante, `RNF-EF-02` técnicamente no
  se rompe), pero **la mayoría de las 20 consultas no recibieron una respuesta real del LLM**,
  solo la derivación.
- Hallazgo secundario: Mailtrap (plan gratis) rechazó los avisos de derivación por límite de
  correos/segundo bajo esta carga — esperado en sandbox, no ocurre con una cuenta de envío
  de producción real.

### Decisión actualizada

Mientras Groq mantenga pausada la alta al plan Developer, la vía 2 de la decisión original
(arriba) no es ejecutable. Opciones reales disponibles ahora mismo, en orden de prioridad:

1. **Monitorear la reapertura** del plan Developer de Groq (histórico: estas pausas suelen
   ser temporales) y activar en cuanto esté disponible — sigue siendo la opción más
   económica y ya compatible con el código sin cambios.
2. **Configurar una clave real y válida de OpenAI** en `.env` (`LLM_2` en la cadena de
   respaldo) como proveedor de pago alterno mientras Groq Developer siga cerrado. El código
   del fallback ya funciona — el fallo de la prueba fue solo por la clave inválida, no por un
   bug.
3. **Instalar un modelo real en Ollama local** (`LLM_3`) como tercera capa de respaldo sin
   costo por token, útil sobre todo en desarrollo. Evaluado para la PC de desarrollo
   (16 GB RAM, GPU NVIDIA GTX 1050 con 4 GB VRAM): un modelo de 3B (ej. `llama3.2:3b` o
   `qwen2.5:3b`) cabe completo en la VRAM dedicada sin competir con la RAM del sistema, que
   ya está ajustada (~3.7 GB libres con el resto de los servicios corriendo). Un modelo de
   7-8B se saldría de la VRAM y empezaría a usar RAM compartida, con riesgo de lentitud.
   *Decisión: no se instala por ahora — no es urgente y el esfuerzo de desarrollo no se
   justifica mientras el plan gratis de Groq baste para el desarrollo normal.*
4. **Reducir la escala del piloto** (menos estudiantes/consultas simultáneas) para quedarse
   dentro del límite del plan gratis de Groq si la fecha del piloto (actividad 6) llega
   antes de que se resuelva cualquiera de los puntos anteriores.

## Para la documentación final de la residencia

Esta decisión y sus números respaldan/corrigen lo siguiente del protocolo:
- Justifica por qué `RNF-EF-02` (20 consultas concurrentes) es una meta razonable y
  alcanzable con el proveedor elegido, siempre que se use el plan Developer en producción
  **o un respaldo de pago equivalente mientras Groq Developer no esté disponible para
  altas nuevas**.
- El costo estimado por semestre contradice y reemplaza el supuesto original de
  "$15-30 USD" del anteproyecto, que no explicaba su origen: con datos reales, el costo es
  del mismo orden de magnitud pero ahora está justificado con cifras verificables de la
  documentación oficial del proveedor, no como una estimación sin respaldo.
- Documenta un caso real de degradación elegante ante fallo simultáneo de las tres capas de
  respaldo configuradas, evidencia útil para la sección de pruebas no funcionales de la
  residencia.
</content>

## Fila por límite de uso y segundo proveedor gratuito (3 de noviembre de 2026)

Para el piloto con el plan gratuito de Groq (~3 consultas por minuto), una consulta ya no
se deriva cuando Groq responde `429` (límite de uso).

### Qué pasa ahora con un `429`

1. **No se reintenta a ciegas.** Antes se esperaba 1, 2 y 4 s y se volvía a pedir. Ahora se
   lee cuánto pide esperar el proveedor: `retry-after` (segundos) o, si no viene,
   `x-ratelimit-reset-tokens` / `x-ratelimit-reset-requests` del límite que se agotó (formato
   `7.66s`, `2m59.56s`). Ese proveedor se salta durante ese tiempo y se pasa al siguiente
   de la cadena.
2. **Si todos están en su límite, la consulta espera en una fila** (`LLM_COLA_*`). El
   estudiante recibe al momento: *"Hay muchas consultas en este momento: tu consulta está
   en fila y te respondo en un momento. Si en 2 minutos no te llega la respuesta, vuelve a
   enviar tu pregunta."* Mientras espera sigue viendo "escribiendo…". La respuesta llega en
   otro mensaje, dividida si es larga y con los botones de calificación al final, como
   siempre.
3. **Las consultas pasan de una en una y en orden de llegada.** Si pasaran todas juntas en
   cuanto el proveedor se libera, volverían a chocar con el límite. Si ya hay fila, quien
   llega se forma aunque el proveedor esté libre.
4. **La espera cuenta dentro de los 80 s** (`LLM_TOTAL_BUDGET`, RNF-FI-02). El tope es lo
   que el estudiante espera en total; darle un tiempo aparte a la fila lo rompería. Al salir
   de la fila deben quedar al menos `LLM_COLA_RESERVA_SEG` (15 s) para generar; si no
   quedan, es mejor decir "alta demanda" de una vez. El límite por minuto de Groq se repone
   en 60 s o menos, así que 60 s de espera máxima alcanzan para el caso normal.
5. **Alta demanda** (motivo `high_demand`): la fila está llena, la espera venció o el
   proveedor pide esperar más de lo que cabe (p. ej. se acabó la cuota **diaria** y Groq
   pide horas). El estudiante recibe *"En este momento hay alta demanda y no pude procesar
   tu consulta. Intenta de nuevo en unos minutos."* No se le deriva a la coordinación ni se
   avisa a la coordinadora.
6. **Falla técnica real** (motivo `llm_unavailable`, como antes): todos los proveedores
   fallan sin que ninguno esté en su límite (caído, timeout, clave inválida, o un `429` de
   OpenAI por falta de saldo). Se conservan el mensaje de antes y el aviso inmediato,
   limitado a uno cada `AVISOS_FALLA_INTERVALO_MIN`.

### Por qué distinguir alta demanda de falla técnica

El límite de un aviso cada 30 min ya evitaba llenar el buzón. Pero con el plan gratuito el
`429` es lo **esperado** en las horas pico: un aviso de "falla técnica" cada media hora por
algo que se resuelve solo enseñaría a la coordinadora a ignorarlo justo cuando haya una
falla de verdad.

Por eso la alta demanda no se avisa. Queda en `metricas.motivo_derivacion` como
`high_demand`, y el panel la muestra como "Alta demanda (límite de uso del servicio de IA)",
aparte de "Servicio de IA no disponible (falla técnica)". Así la coordinadora ve cuántas
consultas se perdieron por saturación, que es el dato que justifica el plan Developer.

Ojo: como ya pasaba con `llm_unavailable`, estas consultas cuentan en la **tasa de
derivación** del panel (resultado `derived`), aunque no se derivaron a nadie.

### Si el bot se reinicia

La fila vive en memoria: una consulta que estaba esperando se pierde. No recibe respuesta
ni queda en `metricas`, y su pregunta queda en `mensajes` sin respuesta. Por eso el aviso de
fila le dice al estudiante que reenvíe su pregunta si en 2 minutos no le llega nada. Como la
espera nunca pasa de los 80 s del tope, a los 2 minutos ya es seguro que se perdió.

### Hallazgo: el bot atendía un mensaje a la vez

`python-telegram-bot` atiende las actualizaciones **de una en una** si no se le indica otra
cosa (`concurrent_updates`, por defecto 1). `asyncio.to_thread` evitaba congelar el bot,
pero no hacía que atendiera a dos estudiantes a la vez. Con una fila de hasta 60 s, eso
habría detenido a todos los demás.

Ahora atiende hasta `BOT_CONCURRENCIA` (20, RNF-EF-02) mensajes a la vez, con un grupo de
hilos del mismo tamaño. Los mensajes de un **mismo** estudiante siguen en orden (un candado
por estudiante), para que su historial no se mezcle.

### Segundo proveedor gratuito: investigación

Revisado el 3 de noviembre de 2026 en la documentación oficial de cada proveedor. "No
confirmado" significa que la página oficial no lo dice.

| Proveedor | Límites gratis (documentación oficial) | ¿Tarjeta? | ¿Entrena con los datos? | Fuente |
|---|---|---|---|---|
| **Cloudflare Workers AI** (`@cf/openai/gpt-oss-120b`) | 10,000 neuronas/día, se reinician a las 00:00 UTC. gpt-oss-120b cuesta 31,818 neuronas por millón de tokens de entrada y 68,182 por millón de salida: ~90 por consulta (~2,000 + ~400 tokens), unas **100 consultas/día**. 300 peticiones/min para generación de texto. Al acabarse la cuota diaria responde 429 (código 3036). | No confirmado en la documentación oficial (fuentes de terceros dicen que no) | **No**: "Cloudflare does not use Customer Content processed by Workers AI to train AI models" | [pricing](https://developers.cloudflare.com/workers-ai/platform/pricing/), [limits](https://developers.cloudflare.com/workers-ai/platform/limits/), [modelo](https://developers.cloudflare.com/workers-ai/models/gpt-oss-120b/), [data usage](https://developers.cloudflare.com/workers-ai/platform/data-usage/), [OpenAI compat](https://developers.cloudflare.com/workers-ai/configuration/open-ai-compatibility/), [errores](https://developers.cloudflare.com/workers-ai/platform/errors/), [modelos que pasaron a pago, 28-jul-2026](https://developers.cloudflare.com/changelog/post/2026-07-28-models-require-workers-paid/) (gpt-oss-120b no está en la lista) |
| **SambaNova Cloud** (`gpt-oss-120b`, `Meta-Llama-3.3-70B-Instruct`, `DeepSeek-V3.1`) | 20 peticiones/min, **20 peticiones/día**, 200,000 tokens/día por modelo | No: el plan gratis aplica justo cuando no hay método de pago | No confirmado | [rate limits](https://docs.sambanova.ai/docs/en/models/rate-limits) |
| **Cerebras** (`gpt-oss-120b`) | 5 peticiones/min, 30K tokens/min sin caché, 1M tokens/hora, 1M/día | **Sí**: crédito único de 5 USD al registrar un método de pago; vence a los 30 días y después se pausa el acceso | No confirmado en la documentación oficial | [rate limits](https://inference-docs.cerebras.ai/support/rate-limits), [pricing](https://www.cerebras.ai/pricing) |
| **Google Gemini API** (`gemini-3.8-flash`, endpoint compatible con OpenAI) | **No publica** cifras del plan gratis: dice que se ven en AI Studio y que "no están garantizados" | No confirmado | **Sí** en el plan gratis: usa lo enviado y lo generado para mejorar sus productos, y revisores humanos pueden leerlo; pide no enviar información personal. Mayores de 18 años. | [rate limits](https://aistudio.google.com/docs/rate-limits), [términos, vigentes desde el 23-mar-2026](https://aistudio.google.com/docs/terms), [OpenAI compat](https://aistudio.google.com/docs/openai) |
| **OpenRouter** (modelos `:free`) | 20 peticiones/min; **50/día** con menos de 10 créditos comprados (1,000/día con 10 o más). Hoy **no hay** `gpt-oss-120b:free` (lista pública `/api/v1/models`, 3-nov-2026). | No para 50/día | Depende del proveedor detrás; hay un ajuste para no usar proveedores que entrenan | [limits](https://openrouter.ai/docs/api-reference/limits), [provider logging](https://openrouter.ai/docs/guides/privacy/provider-logging) |
| **Mistral La Plateforme** | La documentación oficial no publica los límites del plan gratis. Fuentes de terceros hablan de ~1 petición/s y verificación por teléfono: **no confirmado**. | No confirmado | No confirmado | [models](https://docs.mistral.ai/getting-started/models/) |
| **GitHub Models** | **Retirado** el 30-jul-2026 | — | — | [GitHub Docs](https://docs.github.com/en/github-models/use-github-models/prototyping-with-ai-models) |

Para comparar, Groq ([rate limits](https://console.groq.com/docs/rate-limits), revisado el
mismo día): la tabla de la página ahora dice que sus cifras para `openai/gpt-oss-120b` (30
RPM, 1K RPD, 8K TPM, 200K TPD) son los límites base del plan **Developer**, y que los de cada
cuenta se ven en la consola. Las cifras del plan gratis de esta cuenta (8,000 TPM, 200,000
TPD) siguen siendo las que se midieron el 30 de septiembre. La misma página documenta
`retry-after` (solo en 429) y los encabezados `x-ratelimit-*`, que el código ya lee.

### Recomendación: Cloudflare Workers AI como `LLM_2`

1. **Es el mismo modelo** (`gpt-oss-120b`) que usa Groq. La validación por afirmación, las
   reglas del prompt y el reintento por NO_INFO se calibraron con ese modelo. Con otro
   (Gemini, Llama) habría que repetir la evaluación de fundamentación antes del piloto.
   Aun así, la validación aplica igual a cualquier proveedor: está en el pipeline, no en el
   adaptador, y la prueba `test_validacion_por_afirmacion_aplica_igual_con_el_proveedor_de_respaldo`
   lo cubre.
2. **No entrena con los datos**, y lo dice de forma explícita. Gemini gratis sí los usa y
   pueden leerlos personas: no conviene con preguntas de estudiantes.
3. **Unas 100 consultas al día de más** y 300 peticiones/min, la mayor capacidad entre los
   gratuitos verificados. SambaNova da 20 peticiones/día; OpenRouter, 50/día y sin gpt-oss
   gratis; Cerebras pide tarjeta y su crédito vence en 30 días.
4. Cuando se le acaba la cuota diaria responde `429`. El bot lo trata como límite de uso:
   lo salta y, si Groq también está en su límite, la consulta va a la fila y termina, si
   acaso, en "alta demanda", nunca en "falla técnica".

Pendiente de confirmar al activarlo:

- que el plan Workers Free no pida tarjeta (la documentación oficial no lo dice);
- que el endpoint compatible con OpenAI acepte `reasoning_effort`. La página del modelo
  documenta los niveles low/medium/high, pero no el nombre del parámetro en ese endpoint;
  si responde 400, se deja `LLM_2_EXTRA` vacío.

Se prueba con `python -m app.cli check-llm`, que gasta cuota de Cloudflare y no de Groq si
se corre con solo `LLM_2` configurado.

**Cómo activarlo en el `.env` real.** Hoy `LLM_2` es OpenAI con una clave inválida:

1. Cambia `LLM_2_*` por los valores de Cloudflare de `.env.example`.
2. Mueve OpenAI a `LLM_3`, o déjalo sin `LLM_3_BASE_URL` si no hay clave: una clave
   inválida solo hace perder tiempo en cada consulta.

La cadena acepta hasta 5 proveedores (`LLM_1` a `LLM_5`).
