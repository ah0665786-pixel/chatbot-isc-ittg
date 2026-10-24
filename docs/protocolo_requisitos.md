# Requisitos del protocolo 2601-11 (extraídos del Reporte 38º Seminario)

Fuente: `2601-11_Reporte_38Seminario (2).pdf`, secciones 1.3 (requisitos), 1.3.3 (UC e HU), 1.5 (XP).
Extracto de trabajo para auditar trazabilidad. Los números de tabla son los del protocolo.

**Aviso sobre numeración de HU.** El protocolo trae DOS listas de historias de usuario:
- Lista A (Tablas 1-13, sección 1.2.3): HU-01 a HU-13 (HU-01 enviar consulta, HU-04 métricas, HU-12 actualizar KB, HU-13 reportes).
- Lista B (Tablas 60-74, sección 1.3.3): HU-01 a HU-15 (versión final, con 15 historias).
Esta tabla usa la **Lista B**. El código cita números de ambas listas; hay que unificar (ver HU-04/HU-09/HU-13/HU-14).

Los umbrales y modelos del protocolo (similitud 0.75, text-embedding-3-small, GPT-4o mini temp 0.3, LlamaIndex, RDS) difieren de lo implementado
(umbral 0.80, embeddings locales e5, Groq gpt-oss-120b, pipeline propio, PostgreSQL en Docker).

## Requisitos de información (RF-I)
| Código | Requisito | Criterio clave |
|---|---|---|
| RF-I-01 | Recepción de consultas en lenguaje natural vía Telegram, sin comandos | Mensaje recibido y encolado |
| RF-I-02 | Recuperar historial de la conversación activa para coherencia | Historial de la sesión disponible para el LLM |
| RF-I-03 | Consulta a la base de conocimiento: búsqueda de similitud, 5 fragmentos, similitud mínima | Hasta 5 fragmentos; mínimo 0.75 en el protocolo |
| RF-I-04 | Registrar cada mensaje y respuesta con métricas: tiempo (s), fragmentos recuperados, similitud máxima, calificación 1-5 | Tablas MENSAJES y METRICAS |
| RF-I-05 | Coordinadora carga PDF/DOCX; se fragmenta, vectoriza e indexa sin tocar código ni reiniciar | Éxito o error con nombre del archivo problemático |

## Requisitos de comportamiento (RF-C)
| Código | Requisito | Criterio clave |
|---|---|---|
| RF-C-01 | PLN: identificar intención y entidades (inscripción, baja, residencia, servicio social, titulación) | Intención y entidades guían la búsqueda (decisión: sin módulo separado) |
| RF-C-02 | Generar respuesta con LLM usando consulta, historial y fragmentos; temperatura 0.3 | Respuesta fundamentada solo en documentos |
| RF-C-03 | Validar respuesta (fundamentada, tono institucional) antes de enviar | Regenerar si falla |
| RF-C-04 | Derivar a atención humana: similitud baja o consulta excepcional (datos personales, valor legal) | Mensaje con instancia y canal |
| RF-C-05 | Disponibilidad 24/7 | Respuesta en ≤30 s |
| RF-C-06 | Gestión de sesiones: id único, fecha inicio/fin, mensajes asociados | Tabla CONVERSACIONES |

## Reglas del negocio (RF-RN)
| Código | Regla |
|---|---|
| RF-RN-01 | No dar calificaciones, kárdex, adeudos, historial individual ni datos del SIIA; orientar al canal institucional |
| RF-RN-02 | Respuestas fundamentadas exclusivamente en documentación oficial vigente |
| RF-RN-03 | Precisión mínima 90% = consultas respondidas correctamente sin derivación, con consultas de prueba definidas por la coordinadora |

## Requisitos no funcionales
| Código | Requisito | Criterio de aceptación |
|---|---|---|
| RNF-FI-01 | Disponibilidad 99% mensual (AWS, respaldos y reinicio automático) | ≤7.2 h de inactividad al mes, monitoreada |
| RNF-FI-02 | Tolerancia a fallos de la API externa: 3 reintentos con espera exponencial | Espera total ≤90 s; aviso al usuario si no hay servicio |
| RNF-FI-03 | Integridad referencial (USUARIOS, CONVERSACIONES, MENSAJES, DOCUMENTOS, METRICAS) y respaldos diarios en S3 | Sin violaciones de integridad; respaldo cada 24 h |
| RNF-US-01 | Interfaz en lenguaje natural | 90% de usuarios nuevos logra una consulta útil sin instrucciones |
| RNF-US-02 | Claridad y tono institucional | 85% califica ≥4 de 5 |
| RNF-US-03 | Multiplataforma vía Telegram | Funciona en Android, iOS, web y escritorio (prueba de integración) |
| RNF-US-04 | Coordinadora actualiza la KB sin conocimientos técnicos | Carga de documento ≤50 págs en ≤10 min; confirmación visible de éxito o error |
| RNF-SE-01 | Autenticación por ID de Telegram, registro en USUARIOS | Usuario no registrado no genera respuesta del LLM; intentos no autorizados en logs |
| RNF-SE-02 | No solicitar ni guardar datos personales sensibles; cifrado en reposo (RDS en el protocolo) | Sin campos sensibles en tablas; cifrado verificable |
| RNF-SE-03 | HTTPS/TLS; claves y tokens en variables de entorno, nunca en el código | Sin credenciales en texto plano ni en repositorios |
| RNF-SE-04 | Protección contra inyección de prompts | 20 casos rechazados o neutralizados |
| RNF-EF-01 | Respuesta ≤30 s | 95% de consultas ≤30 s, medido en METRICAS |
| RNF-EF-02 | 20 consultas concurrentes sin degradación | Promedio ≤30 s y sin errores 5xx |
| RNF-EF-03 | Búsqueda vectorial ≤2 s, índice en RAM | ≤2 s en el 99%, registrado en logs |
| RNF-PO-01 | Despliegue en EC2 (t3.medium dev, t3.large prod) con Docker | Despliegue con solo Dockerfile y docker-compose.yml |
| RNF-PO-02 | Escalable a otras carreras cambiando KB y configuración | Solo reindexar y cambiar variables |
| RNF-PO-03 | Backend desacoplado del canal; Telegram se comunica por API REST de FastAPI | Capa de interfaz sin lógica de negocio ni acceso directo a BD o motor IA |

## Historias de usuario (Lista B, Tablas 60-74) y criterios de aceptación
| HU | Rol y acción | Criterios de aceptación |
|---|---|---|
| HU-01 | Estudiante: autenticarme por Telegram | CA-01 registro automático por ID; CA-02 sin contraseñas; CA-03 accesos no autorizados en logs; CA-04 no se vuelve a registrar |
| HU-02 | Estudiante: enviar consultas en lenguaje natural | CA-01 texto libre; CA-02 <30 s; CA-03 indicador de escritura; CA-04 si supera el límite de caracteres, pedir dividirlo |
| HU-03 | Estudiante: consultar requisitos de trámites | CA-01 fundamentada en documentos oficiales; CA-02 pasos y documentos del trámite; CA-03 datos personales se derivan; CA-04 <30 s |
| HU-04 | Estudiante: preguntas de seguimiento | CA-01 historial desde la BD; CA-02 respuestas coherentes; CA-03 sesión expirada abre una nueva; CA-04 historial en el contexto del LLM |
| HU-05 | Sistema: identificar intención y entidades | CA-01 clasifica inscripción, baja, titulación, residencia, servicio social; CA-02 entidades disponibles para RAG; CA-03 fuera de dominio deriva; CA-04 PLN añade ≤5 s |
| HU-06 | Estudiante: respuestas contextualizadas y precisas | CA-01 solo fragmentos con similitud mínima; CA-02 tono formal; CA-03 extensión para móvil; CA-04 temperatura 0.3 |
| HU-07 | Sistema: validar respuesta antes de enviar | CA-01 rechazar si no está fundamentada; CA-02 regenerar; CA-03 si falla el 2.º intento, derivar; CA-04 resultado registrado |
| HU-08 | Estudiante: orientación cuando no hay respuesta | CA-01 indica instancia (Coordinación o Servicios Escolares); CA-02 canal y horario; CA-03 deriva si similitud baja o fuera de dominio; CA-04 registro con indicador de derivación |
| HU-09 | Coordinadora: registrar y consultar métricas | CA-01 registro automático; CA-02 reportes filtrables por periodo; CA-03 tasa de derivación visible; CA-04 último reporte si no hay datos |
| HU-10 | Sistema: gestionar sesiones | CA-01 sesión nueva al primer mensaje; CA-02 mensajes con el mismo id; CA-03 cierre por inactividad configurable; CA-04 reanudar crea sesión nueva |
| HU-11 | Coordinadora: recibir escalamientos con contexto | CA-01 contexto por correo institucional; CA-02 horario y correo al estudiante; CA-03 registro en METRICAS; CA-04 no escalar lo resoluble |
| HU-12 | Estudiante: acceso desde cualquier dispositivo | CA-01 Android, iOS, web, escritorio; CA-02 sin instalar nada; CA-03 24/7; CA-04 <30 s |
| HU-13 | Coordinadora: reportes de desempeño | CA-01 total de consultas, precisión, tiempo promedio, derivación y tipos de consulta más frecuentes; CA-02 filtro por fechas; CA-03 precisión = % resueltas sin derivación; CA-04 exportable |
| HU-14 | Coordinadora: actualizar base de conocimiento | CA-01 PDF y DOCX; CA-02 fragmenta y vectoriza solo; CA-03 índice se recarga de inmediato; CA-04 error con nombre de archivo; CA-05 sin tocar el código |
| HU-15 | Coordinadora: desempeño histórico | CA-01 métricas desde el despliegue; CA-02 agrupadas por semana, mes o ciclo; CA-03 tipos de consulta más frecuentes y áreas con más derivación; CA-04 indicar periodo mínimo si faltan datos |

## Casos de uso (UC-01 a UC-15)
UC-01 Autenticar estudiante · UC-02 Enviar consulta · UC-03 Consultar trámite · UC-04 Conversación contextual ·
UC-05 Orientar a área responsable · UC-06 Notificar respuesta · UC-07 Procesar con PLN · UC-08 Buscar en FAISS (RAG) ·
UC-09 Generar respuesta (LLM) · UC-10 Registrar interacciones en PostgreSQL · UC-11 Gestionar sesión ·
UC-12 Escalar a coordinadora (correo con contexto) · UC-13 Validar respuesta · UC-14 Consultar métricas · UC-15 Actualizar base FAISS.

## Evaluación y criterio de éxito (1.5 XP y anteproyecto)
- Precisión ≥90% al cierre de la iteración 6, medida como consultas respondidas correctamente sin derivación, con criterios de la coordinadora (RF-RN-03).
- Tiempo de respuesta <30 s; satisfacción medida con encuestas; pruebas de carga con 20 usuarios; 20 casos de inyección de prompts.
- Iteración 6 (sem 11-12): HU-15, carga, inyección, ajuste de umbral, despliegue en AWS t3.large.
