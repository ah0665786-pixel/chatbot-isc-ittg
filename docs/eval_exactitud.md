# Evaluación de exactitud con preguntas de estudiantes

Conjunto de 48 casos (`data/eval_exactitud.jsonl`) para medir si el chatbot responde bien
las preguntas de los estudiantes y deriva las que no debe contestar. Lo corre el script
`app/eval_exactitud.py`.

**Estado: todos los casos están "pendiente de validación por la coordinadora".** Hasta que
ella los revise, un resultado de esta evaluación mide el chatbot contra lo que *nosotros*
leímos en los documentos, no contra el criterio de la coordinación.

## Qué trae cada caso

| Campo | Contenido |
|---|---|
| `q` | Pregunta escrita como la haría un estudiante: informal, sin "¿", a veces con faltas ("q", "resi", "especial"), sin copiar las palabras del documento |
| `esperado` | `responder` (40 casos) o `derivar` (8 casos) |
| `respuesta_esperada` | Lo que debería contestar, sacado de los PDF de `data/docs` (no del bot) |
| `cita`, `documento`, `pagina`, `numeral` | Cita textual, archivo y **página del PDF** (la misma numeración que cita el bot; no es el número impreso en la hoja) |
| `hechos.incluye` / `hechos.no_incluye` | Expresiones regulares sobre la respuesta en minúsculas y sin acentos: lo que debe decir y lo que no (p. ej. "20 días naturales" cuando el documento dice hábiles) |
| `motivo_esperado` | En los casos a derivar: `out_of_scope` (datos personales o documentos oficiales) o `low_similarity`/`not_in_context` (no está en los documentos) |
| `nota` | Diferencias entre documentos que la coordinadora debe resolver |
| `validacion` | `pendiente de validación por la coordinadora` |

Temas: plan de estudios (6), residencia (5), servicio social (6), titulación (5), bajas (6),
traslados y convalidación (6), actividades complementarias (3), permisos y evaluaciones (3), y
8 casos a derivar: 4 de datos personales o documentos oficiales, y 4 fuera de los documentos
(costos, horarios, permisos para faltar y una pregunta fuera del tema).

Las citas se transcribieron de los PDF; solo se corrigieron los espacios que la extracción de
texto mete dentro de las palabras. `--verificar` comprueba que cada cita esté en su página.

## Documentos que no coinciden (para la coordinadora)

Donde el Manual de Lineamientos del TecNM (2015) y un lineamiento suelto (2010-2012) dicen
cosas distintas, el caso usa el Manual y lo anota en `nota`:

- **Convalidación (`conv-01`).** El lineamiento de 2010 dice que "aplica máximo para
  estudiantes en cuarto semestre"; el Manual no lo dice.
- **Cambio de asesor de titulación (`tit-05`).** El lineamiento de 2012 dice que la solicitud
  va a la División de Estudios Profesionales; el Manual dice Comité Académico.
- **Mención honorífica (`tit-04`).** Las alternativas cambiaron entre 2012 y 2015.
- **Servicio social (`ss-01`, `ss-03`).** El 70 % de créditos y el máximo de 2 años vienen del
  lineamiento de 2011. El Manual remite a la "normativa vigente" y solo dice "no menor de seis
  meses".

## Comandos

Ninguno llama al modelo, salvo `--correr`:

```
python -m app.eval_exactitud --verificar      # citas contra los PDF y hechos contra la respuesta esperada
python -m app.eval_exactitud --estimar        # tokens que gastaría con Groq
python -m app.eval_exactitud --revision docs/eval_exactitud_revision.docx
python -m app.eval_exactitud --correr --si-gastar --env C:/ChatbotSis/chatbot-isc/.env \
    --kb C:/ChatbotSis/chatbot-isc/data/kb --out data/eval_exactitud_FECHA.jsonl
python -m app.eval_exactitud --resumen data/eval_exactitud_FECHA.jsonl
```

Cómo funciona `--correr`:

- Usa el pipeline completo (búsqueda real + Groq) **sin base de datos**: no deja métricas en
  la base real.
- Espera 22 s por llamada para quedar en ~3 por minuto.
- Guarda cada resultado al llegar: si se corta, el mismo comando sigue con los que faltan.
- Se detiene si Groq avisa del tope diario o al gastar `--tope-tokens` (90,000 por defecto,
  para dejarle al bot más de la mitad de la cuota del día).

`--resumen` reporta:

- **acierto**, sobre los casos a responder: responde, trae todos los hechos y no añade nada
  falso;
- **derivación correcta**, sobre los casos a derivar;
- **errores**, sobre todos: dice algo que el documento no dice, o responde lo que debía derivar;
- **incompletas**: sin nada falso, pero le falta un hecho;
- **derivó de más**: tenía respuesta y derivó;
- **inconclusos**: falló el proveedor o hubo alta demanda; hay que repetirlos.

Además escribe `..._revision.csv` con los **dudosos** para revisarlos a mano: los incompletos,
los errores, los que derivaron por un motivo distinto al esperado y los que no citan el
documento esperado.

## Tokens estimados con Groq (sin correrlo)

`--estimar` (3 de noviembre de 2026) arma localmente el prompt que el bot mandaría en cada caso
(búsqueda real con la base de conocimiento actual, sin llamar al modelo). Lo calibra con las 130
consultas ya medidas con Groq en `data/fundamentacion_*.jsonl`:

| Dato | Valor |
|---|---|
| Casos que llegan al modelo | 45 de 48 (3 los bloquea el filtro antes) |
| Calibración (88 consultas de una llamada) | 0.234 tokens por carácter de prompt + 256 fijos |
| Regeneraciones (130 consultas) | 1.35 llamadas por consulta; factor 1.37 sobre una llamada |
| **Tokens esperados** | **~127,000** (64 % del tope diario de 200,000) |
| Mínimo (una llamada por caso) | ~93,000 |
| Peor caso (3 llamadas cada uno) | ~306,000 (no cabe en un día) |
| Llamadas y tiempo | ~61 llamadas, ~22 min de espera entre llamadas |

Con `--tope-tokens 90000`, por defecto, la corrida se reparte en **dos días**. La cuota de
Groq es la misma que usa el bot con los estudiantes, así que conviene correrla fuera del
horario del piloto. **No se ha corrido.**

## Hallazgos sin gastar cuota

Al estimar se vio qué casos llegan al modelo:

- `perm-02`: "en cuanto tiempo me tiene q dar el profe mis calificaciones de los parciales"
  **no llega al modelo**. El filtro de datos personales la bloquea por "calificaciones",
  aunque la respuesta está en el Manual (pág. 44, 5.4.5.8: cinco días hábiles). Falso
  positivo del filtro.
- `der-01` ("cuanto saque en calculo integral?") y `der-03` ("me generas una constancia de
  estudios") **sí llegan al modelo**: el filtro no las reconoce como datos personales o
  documento oficial. Dependen de que el modelo diga NO_INFO.
- `der-05`, `der-06`, `der-07` y `der-08` (costos, horario, permiso para faltar, laptop)
  pasan el umbral de similitud 0.80 y llegan al modelo: también dependen del NO_INFO.

La primera corrida dirá si estos casos terminan bien. Los ajustes al filtro son una tarea
aparte.

## Archivo para la coordinadora

`docs/eval_exactitud_revision.docx` (para leer o imprimir, hoja horizontal) y
`docs/eval_exactitud_revision.csv` (para Excel). Cada fila trae:

- el tema y la pregunta;
- lo que el bot debe responder, o "DEBE DERIVAR";
- la cita textual;
- el documento y la página del PDF;
- la columna **"¿Correcta o corregir?"** y una columna de comentarios.

Cuando la coordinadora los devuelva, se corrigen los casos marcados "corregir" en
`data/eval_exactitud.jsonl`. Luego `validacion` pasa a `validado por la coordinadora (FECHA)`,
y se vuelven a generar los dos archivos con `--revision`.
