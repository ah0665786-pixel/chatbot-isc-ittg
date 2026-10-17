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
