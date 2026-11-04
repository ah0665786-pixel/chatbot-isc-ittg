"""Fila por límite de uso (HTTP 429), segundo proveedor y "alta demanda".

Todo sin red (httpx.MockTransport) y con reloj y esperas simulados: la fila
"espera" adelantando un reloj falso, así que las pruebas no tardan lo que
tardaría el proveedor de verdad.
"""
from __future__ import annotations

import threading
import time
from dataclasses import replace

import httpx
import pytest

from app.adapters.db_sqlite import SqliteRepository
from app.adapters.llm_openai_compat import (FallbackLLM, HighDemand, LLMUnavailable,
                                            OpenAICompatLLM, QueuePolicy, RateLimited,
                                            parse_duration, queue_listener,
                                            retry_after_seconds)
from app.config import Settings
from app.domain import REASON_LABELS, Outcome, Reason
from app.notify import should_notify
from app.rag.messages import HIGH_DEMAND_MESSAGE
from app.rag.pipeline import RagPipeline
from app.service import ConversationService

MSG = [{"role": "user", "content": "hola"}]
GOOD = ("La baja temporal se solicita ante la Coordinación con el formato de baja temporal "
        "dentro de los primeros 20 días hábiles del semestre. (Reglamento Escolar.txt)")


class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.waits: list[float] = []

    def __call__(self):
        return self.now

    def sleep(self, s):
        self.now += s

    def wait(self, cond, seconds):  # lo que usa la fila en lugar de Condition.wait
        self.waits.append(round(seconds, 3))
        self.now += seconds


def ok(text="listo", **headers):
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]},
                          headers=headers)


def limited(retry_after="3", **headers):
    h = {"retry-after": retry_after} if retry_after is not None else {}
    h.update(headers)
    return httpx.Response(429, json={"error": {"message": "Rate limit reached for tokens per "
                                               "minute (TPM)"}}, headers=h)


class Script:
    """Proveedor guionado: devuelve las respuestas en orden (la última se repite)
    y cuenta las peticiones; cada petición "tarda" `cost` segundos del reloj."""

    def __init__(self, clock, *responses, cost=0.0):
        self.clock, self.responses, self.cost = clock, list(responses), cost
        self.calls = 0

    def __call__(self, req):
        self.calls += 1
        self.clock.now += self.cost
        r = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        # una copia nueva cada vez: la misma respuesta puede repetirse
        return httpx.Response(r.status_code, headers=r.headers, content=r.content)


def provider(clock, script, name="groq", retries=3):
    return OpenAICompatLLM(name, "http://x/v1", "k", "m", max_retries=retries,
                           transport=httpx.MockTransport(script), sleep=clock.sleep,
                           clock=clock)


def chain(clock, *providers, **policy):
    return FallbackLLM(list(providers), total_budget=80, clock=clock,
                       queue=QueuePolicy(**{"max_len": 5, "max_wait": 60, "reserve": 15,
                                            **policy}),
                       wait=clock.wait)


class Listener:
    def __init__(self):
        self.calls = []

    def __enter__(self):
        self.token = queue_listener.set(self.calls.append)
        return self

    def __exit__(self, *exc):
        queue_listener.reset(self.token)


# ---- encabezados del proveedor -------------------------------------------------
@pytest.mark.parametrize("value,secs", [
    ("2", 2.0), ("7.66s", 7.66), ("2m59.56s", 179.56), ("1h2m3s", 3723.0), ("120ms", 0.12),
    ("", None), ("pronto", None),
])
def test_parse_duration_formato_groq_y_openai(value, secs):
    assert parse_duration(value) == (pytest.approx(secs) if secs is not None else None)


def test_retry_after_tiene_prioridad_sobre_x_ratelimit():
    h = {"Retry-After": "4", "x-ratelimit-reset-tokens": "30s",
         "x-ratelimit-remaining-tokens": "0"}
    assert retry_after_seconds(h) == 4


def test_sin_retry_after_usa_el_reinicio_del_limite_agotado():
    # Groq: -tokens es por minuto, -requests es por día. Se agotaron los tokens.
    h = {"x-ratelimit-reset-tokens": "7.66s", "x-ratelimit-remaining-tokens": "0",
         "x-ratelimit-reset-requests": "2m59.56s", "x-ratelimit-remaining-requests": "14370"}
    assert retry_after_seconds(h) == pytest.approx(7.66)
    # Se agotó el límite diario de peticiones: hay que esperar a su reinicio
    h2 = {**h, "x-ratelimit-remaining-tokens": "500", "x-ratelimit-remaining-requests": "0"}
    assert retry_after_seconds(h2) == pytest.approx(179.56)


def test_retry_after_como_fecha_http():
    assert retry_after_seconds({"retry-after": "Wed, 21 Oct 2015 07:28:10 GMT"},
                               now=lambda: 1445412480.0) == pytest.approx(10)


def test_429_sin_encabezados_usa_espera_por_defecto():
    assert retry_after_seconds({}) == 5.0


def test_429_se_reporta_al_momento_sin_reintentos_a_ciegas():
    clock = FakeClock()
    s = Script(clock, limited("3"))
    with pytest.raises(RateLimited) as e:
        provider(clock, s).chat(MSG)
    assert s.calls == 1 and e.value.retry_after == 3 and "429" in str(e.value)
    assert clock.now == 1000  # no gastó el tope durmiendo


def test_429_sin_saldo_es_falla_tecnica_no_limite_de_uso():
    clock = FakeClock()
    s = Script(clock, httpx.Response(429, json={"error": {"code": "insufficient_quota"}}))
    with pytest.raises(LLMUnavailable) as e:
        chain(clock, provider(clock, s, "openai")).chat(MSG)
    assert not isinstance(e.value, HighDemand)


# ---- fila --------------------------------------------------------------------------
def test_429_espera_en_fila_lo_que_pide_el_proveedor_y_luego_responde():
    clock = FakeClock()
    s = Script(clock, limited("3"), ok("respuesta tras la fila"))
    llm = chain(clock, provider(clock, s))
    with Listener() as heard:
        r = llm.chat(MSG)
    assert r.text == "respuesta tras la fila" and s.calls == 2
    assert clock.waits == [3.0] and clock.now == 1003  # esperó justo lo que pidió
    assert len(heard.calls) == 1 and heard.calls[0].position == 1
    assert heard.calls[0].estimated_wait == 3
    assert llm.queued == 0  # salió de la fila


def test_usa_x_ratelimit_reset_tokens_de_groq_si_no_hay_retry_after():
    clock = FakeClock()
    s = Script(clock, limited(None, **{"x-ratelimit-reset-tokens": "7.66s",
                                       "x-ratelimit-remaining-tokens": "0"}), ok())
    chain(clock, provider(clock, s)).chat(MSG)
    assert clock.waits == [pytest.approx(7.66)]


def test_fila_llena_es_alta_demanda_sin_peticiones_extra():
    clock = FakeClock()
    s = Script(clock, limited("10"), ok())
    llm = chain(clock, provider(clock, s), max_len=1)
    with pytest.raises(RateLimited):  # el proveedor ya está en su límite
        llm.providers[0].chat(MSG)
    llm._cooling["groq"] = clock() + 10
    other = llm._join(clock() + 80)[0]  # alguien más ya está esperando
    with Listener() as heard, pytest.raises(HighDemand) as e:
        llm.chat(MSG)
    assert "fila llena" in str(e.value)
    assert s.calls == 1 and heard.calls == [] and clock.now == 1000  # sin esperar ni pedir
    llm._leave(other)


def test_espera_vencida_es_alta_demanda():
    clock = FakeClock()
    s = Script(clock, limited("10"))  # el proveedor sigue en su límite
    llm = chain(clock, provider(clock, s), max_wait=25)
    with Listener() as heard, pytest.raises(HighDemand) as e:
        llm.chat(MSG)
    assert "vencida" in str(e.value)
    assert clock.now - 1000 <= 25 + 0.001  # nunca esperó más de LLM_COLA_ESPERA_MAX_SEG
    assert len(heard.calls) == 1  # un solo aviso aunque volvió a la fila
    assert s.calls == 3  # 0 s, 10 s, 20 s; a los 25 s se rinde sin otra petición
    assert llm.queued == 0


def test_si_la_espera_pedida_no_cabe_es_alta_demanda_al_momento():
    """p. ej. se agotó el límite diario (Groq pide esperar horas)."""
    clock = FakeClock()
    s = Script(clock, limited("3600"))
    with Listener() as heard, pytest.raises(HighDemand):
        chain(clock, provider(clock, s)).chat(MSG)
    assert clock.now == 1000 and heard.calls == []  # ni espera ni aviso de fila


def test_fila_desactivada_con_cero():
    clock = FakeClock()
    with pytest.raises(HighDemand):
        chain(clock, provider(clock, Script(clock, limited("2"))), max_len=0).chat(MSG)
    assert clock.now == 1000


def test_quien_llega_despues_se_forma_aunque_el_proveedor_este_libre():
    clock = FakeClock()
    s = Script(clock, ok())
    llm = chain(clock, provider(clock, s))
    first = llm._join(clock() + 80)[0]  # alguien espera desde antes
    with Listener() as heard, pytest.raises(HighDemand):
        llm.chat(MSG)  # no le toca: es la segunda de la fila
    assert s.calls == 0 and len(heard.calls) == 1 and heard.calls[0].position == 2
    llm._leave(first)


# ---- segundo proveedor --------------------------------------------------------------
def test_429_del_primero_pasa_al_segundo_sin_fila_y_lo_salta_mientras_se_enfria():
    clock = FakeClock()
    groq = Script(clock, limited("30"))
    respaldo = Script(clock, ok("desde el respaldo"))
    llm = chain(clock, provider(clock, groq, "groq"), provider(clock, respaldo, "cloudflare"))
    with Listener() as heard:
        r1 = llm.chat(MSG)
        r2 = llm.chat(MSG)
    assert (r1.provider, r2.provider) == ("cloudflare", "cloudflare")
    assert groq.calls == 1  # la 2a consulta ya no le pidió a Groq: sigue en su límite
    assert heard.calls == [] and clock.waits == []
    clock.now += 31  # Groq vuelve a aceptar
    groq.responses = [ok("desde groq")]
    assert llm.chat(MSG).provider == "groq"


def test_429_en_uno_y_falla_tecnica_en_otro_espera_al_que_tiene_limite():
    clock = FakeClock()
    groq = Script(clock, limited("4"), ok("groq de vuelta"))
    roto = Script(clock, httpx.Response(401))
    llm = chain(clock, provider(clock, groq, "groq"), provider(clock, roto, "openai"))
    assert llm.chat(MSG).text == "groq de vuelta"
    assert clock.waits == [4.0]


def test_todos_con_falla_tecnica_no_es_alta_demanda():
    clock = FakeClock()
    llm = chain(clock, provider(clock, Script(clock, httpx.Response(401)), "a"),
                provider(clock, Script(clock, httpx.Response(500)), "b", retries=1))
    with pytest.raises(LLMUnavailable) as e:
        llm.chat(MSG)
    assert not isinstance(e.value, HighDemand)


# ---- tope total (RNF-FI-02) ----------------------------------------------------------
def test_la_espera_en_fila_cuenta_dentro_del_tope_y_deja_la_reserva():
    clock = FakeClock()
    s = Script(clock, limited("8"), cost=2.0)  # cada petición tarda 2 s y siempre da 429
    llm = chain(clock, provider(clock, s), max_wait=500, reserve=15)
    with pytest.raises(HighDemand):
        llm.chat(MSG)
    # Con max_wait enorme manda el tope: deja de esperar cuando quedan 15 s
    assert clock.now - 1000 <= 80 - 15 + 2.0 + 0.001


def test_el_deadline_del_pipeline_manda_sobre_la_fila():
    clock = FakeClock()
    s = Script(clock, limited("5"), ok())
    llm = chain(clock, provider(clock, s), reserve=15)
    with pytest.raises(HighDemand):  # quedan 18 s: esperar 5 dejaría 13 < reserva de 15
        llm.chat(MSG, deadline=clock() + 18)
    assert s.calls == 1 and clock.now == 1000


def test_la_peticion_tras_la_fila_recorta_su_timeout_al_tope():
    clock = FakeClock()
    timeouts = []

    def h(req):
        timeouts.append(req.extensions["timeout"]["read"])
        return limited("40") if len(timeouts) == 1 else ok()

    p = OpenAICompatLLM("groq", "http://x/v1", "k", "m", timeout=25,
                        transport=httpx.MockTransport(h), sleep=clock.sleep, clock=clock)
    chain(clock, p, max_wait=60, reserve=15).chat(MSG, deadline=clock() + 60)
    # 60 s de tope - 40 s en la fila: a la petición solo le quedan 20 s
    assert timeouts == [25, pytest.approx(20)]


# ---- varias consultas a la vez (hilos de verdad) --------------------------------------
def test_la_fila_deja_pasar_una_a_la_vez_y_en_orden_de_llegada():
    """Tres consultas topan con el límite al mismo tiempo; al liberarse, pasan
    de una en una (si pasaran juntas volverían a topar con el límite) y en el
    orden en que entraron a la fila."""
    lock = threading.Lock()
    state = {"calls": 0, "active": 0, "peak": 0}
    served: list[str] = []

    def h(req):
        with lock:
            state["calls"] += 1
            n = state["calls"]
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        try:
            if n <= 3:
                return httpx.Response(429, headers={"retry-after": "0.3"})
            time.sleep(0.05)
            return ok(req.content.decode()[-20:])
        finally:
            with lock:
                state["active"] -= 1

    p = OpenAICompatLLM("groq", "http://x/v1", "k", "m", transport=httpx.MockTransport(h))
    llm = FallbackLLM([p], total_budget=20, queue=QueuePolicy(5, 10, 1))
    barrier = threading.Barrier(3)
    order: list[int] = []

    def worker(i):
        barrier.wait()
        time.sleep(0.02 * i)  # llegan en orden 0, 1, 2
        tok = queue_listener.set(lambda info: order.append(i))
        try:
            llm.chat([{"role": "user", "content": f"consulta {i}"}])
            served.append(str(i))
        finally:
            queue_listener.reset(tok)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert served == ["0", "1", "2"] == [str(i) for i in order]
    assert state["calls"] == 6  # 3 con 429 + 3 respuestas, sin 429 de más
    assert llm.queued == 0


# ---- de punta a punta: pipeline y servicio ---------------------------------------------
def _service(kb, cfg, llm, clock):
    pipeline = RagPipeline(cfg, kb.embedder, kb.store, llm)
    pipeline._clock = clock
    return ConversationService(cfg, SqliteRepository(cfg.db_path), pipeline)


def test_servicio_avisa_la_fila_una_vez_y_responde(kb, cfg):
    clock = FakeClock()
    s = Script(clock, limited("3"), ok(GOOD))
    svc = _service(kb, cfg, chain(clock, provider(clock, s)), clock)
    heard = []
    result, _ = svc.handle_message("tg-1", "Ana", "¿Cómo solicito la baja temporal?",
                                   on_queued=heard.append)
    assert result.outcome is Outcome.ANSWERED and len(heard) == 1
    assert queue_listener.get() is None  # no se queda puesto para la siguiente consulta


def test_todos_en_su_limite_alta_demanda_sin_derivar_ni_avisar(kb, cfg):
    cfg = replace(cfg, avisos_resumen_hora="", avisos_modo="vista_previa")  # todo inmediato
    clock = FakeClock()
    groq = Script(clock, limited("3600"))
    respaldo = Script(clock, limited("3600"))
    svc = _service(kb, cfg, chain(clock, provider(clock, groq, "groq"),
                                  provider(clock, respaldo, "cloudflare")), clock)
    submitted = []
    svc.notifier.submit = lambda *a, **k: submitted.append(a)
    for n in range(3):
        result, metric_id = svc.handle_message(f"tg-{n}", "Ana", "¿Cómo solicito la baja temporal?")
        assert result.reason is Reason.HIGH_DEMAND and result.text == HIGH_DEMAND_MESSAGE
        assert "Coordinación" not in result.text and "minutos" in result.text
        assert not should_notify(result)
    assert submitted == []  # ningún aviso a la coordinadora, ni uno por consulta
    assert not list((cfg.outbox_dir).glob("*.eml")) if cfg.outbox_dir.exists() else True
    summary = svc.repo.metrics_summary()
    assert summary.by_reason == {"high_demand": 3}
    assert REASON_LABELS["high_demand"] != REASON_LABELS["llm_unavailable"]


def test_falla_tecnica_real_sigue_siendo_llm_unavailable_y_se_avisa(kb, cfg):
    clock = FakeClock()
    s = Script(clock, httpx.Response(401))
    svc = _service(kb, cfg, chain(clock, provider(clock, s)), clock)
    result, _ = svc.handle_message("tg-1", "Ana", "¿Cómo solicito la baja temporal?")
    assert result.reason is Reason.LLM_UNAVAILABLE and should_notify(result)


def test_validacion_por_afirmacion_aplica_igual_con_el_proveedor_de_respaldo(kb, cfg):
    """El respaldo puede ser otro modelo: su respuesta pasa por la misma
    validación (una cifra inventada se rechaza y se regenera)."""
    clock = FakeClock()
    bad = GOOD.replace("20 días", "45 días")
    groq = Script(clock, limited("3600"))
    respaldo = Script(clock, ok(bad), ok(GOOD))
    llm = chain(clock, provider(clock, groq, "groq"), provider(clock, respaldo, "cloudflare"))
    pipeline = RagPipeline(cfg, kb.embedder, kb.store, llm)
    pipeline._clock = clock
    r = pipeline.answer("¿Cómo solicito la baja temporal?")
    assert r.outcome is Outcome.ANSWERED and r.provider == "cloudflare" and r.attempts == 2
    assert "45" not in r.text and respaldo.calls == 2


def test_config_de_la_fila_desde_el_env(monkeypatch):
    monkeypatch.setenv("LLM_COLA_MAX", "8")
    monkeypatch.setenv("LLM_COLA_ESPERA_MAX_SEG", "45")
    monkeypatch.setenv("LLM_COLA_RESERVA_SEG", "10")
    monkeypatch.setenv("BOT_CONCURRENCIA", "12")
    cfg = Settings.from_env(load_dotenv_file=False)
    assert (cfg.llm_queue_max, cfg.llm_queue_max_wait, cfg.llm_queue_reserve,
            cfg.bot_concurrency) == (8, 45.0, 10.0, 12)
    d = Settings()
    assert (d.llm_queue_max, d.llm_queue_max_wait, d.llm_queue_reserve) == (5, 60.0, 15.0)


def test_la_cadena_acepta_hasta_cinco_proveedores_y_salta_los_incompletos(monkeypatch):
    for i in range(1, 7):
        monkeypatch.setenv(f"LLM_{i}_NAME", f"p{i}")
        monkeypatch.setenv(f"LLM_{i}_BASE_URL", f"http://p{i}/v1")
        monkeypatch.setenv(f"LLM_{i}_MODEL", "m")
    monkeypatch.setenv("LLM_2_BASE_URL", "")  # p. ej. Cloudflare sin activar todavía
    names = [e.name for e in Settings.from_env(load_dotenv_file=False).llm_endpoints]
    assert names == ["p1", "p3", "p4", "p5"]
