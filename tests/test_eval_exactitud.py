"""Conjunto de evaluación de exactitud (data/eval_exactitud.jsonl) y su script.
Nada aquí llama al modelo: la corrida real se prueba con un pipeline simulado."""
from __future__ import annotations

import csv
import json
from collections import Counter
from types import SimpleNamespace

import pytest

from app import eval_exactitud as ev

CASES = ev.load_cases()
PROTOCOL_TOPICS = {"plan_estudios", "residencia", "servicio_social", "titulacion", "bajas",
                   "traslados", "actividades", "permisos"}


# ---- el conjunto --------------------------------------------------------------------------
def test_tamano_temas_y_casos_a_derivar():
    assert 40 <= len(CASES) <= 50
    assert len({c["id"] for c in CASES}) == len(CASES)
    assert PROTOCOL_TOPICS <= {c["tema"] for c in CASES if c["esperado"] == ev.ANSWER}
    derive = [c for c in CASES if c["esperado"] == ev.DERIVE]
    assert 7 <= len(derive) <= 9
    motives = Counter(m for c in derive for m in c["motivo_esperado"])
    assert motives["out_of_scope"] >= 3 and motives["low_similarity"] >= 3  # datos personales y fuera


def test_todos_pendientes_de_validacion_por_la_coordinadora():
    assert {c["validacion"] for c in CASES} == {ev.PENDING}


def test_cada_caso_a_responder_trae_cita_documento_pagina_y_hechos():
    for c in (c for c in CASES if c["esperado"] == ev.ANSWER):
        assert c["cita"] and c["documento"].endswith(".pdf") and c["pagina"] >= 1, c["id"]
        assert c["hechos"]["incluye"] and c["hechos"]["no_incluye"], c["id"]
        assert c["respuesta_esperada"] and c["numeral"], c["id"]


def test_preguntas_informales_y_no_copiadas_del_documento():
    for c in (c for c in CASES if c["esperado"] == ev.ANSWER):
        q = ev._squash(c["q"])
        # ninguna pregunta contiene un tramo de 30 caracteres seguidos de su cita
        assert not any(q[i:i + 30] in ev._squash(c["cita"]) for i in range(max(len(q) - 29, 1))), c["id"]
    informal = [c for c in CASES if not c["q"].startswith("¿")]
    assert len(informal) >= len(CASES) * 0.8  # como escriben en Telegram, sin "¿"


def test_citas_en_su_pagina_y_respuestas_cumplen_sus_hechos():
    """Lee los PDF de data/docs: cada cita está en la página que dice, y la
    respuesta esperada pasa sus propios hechos (si no, el hecho está mal escrito)."""
    assert ev.verify(CASES) == []


def test_verify_detecta_cita_falsa_y_hecho_mal_escrito():
    bad = [dict(c) for c in CASES[:1]]
    bad[0]["cita"] = "El servicio social dura tres años y se paga con beca."
    bad[0]["hechos"] = {"incluye": ["texto que no esta"], "no_incluye": ["("]}
    problems = ev.verify(bad)
    assert any("no aparece" in p for p in problems)
    assert any("no cumple sus hechos" in p for p in problems)
    assert any("regex inválida" in p for p in problems)


EXTRACTO = "Extracto_Manual_Lineamientos_Servicio_Social.docx"
BASE = {"Manual_de_Lineamientos_TecNM.pdf", "Reglamento de Estudiantes TecNM.pdf", EXTRACTO}


def test_ss01_acepta_la_cita_del_extracto_o_la_del_glosario():
    ss01 = next(c for c in CASES if c["id"] == "ss-01")
    assert [x["documento"] for x in ss01["fuentes_aceptadas"]] == [EXTRACTO]
    respuesta = "Debes tener aprobado el 70 % de los créditos de tu plan de estudios."
    for cita in ("(Extracto Manual Lineamientos Servicio Social)",  # como lo citó el bot
                 "(Manual de Lineamientos TecNM, pág. 183)"):
        v = ev.classify(ss01, "answered", "none", f"{respuesta} {cita}")
        assert (v.category, v.doubtful) == ("acierto", False), cita
    v = ev.classify(ss01, "answered", "none", f"{respuesta} (Reglamento de Estudiantes)")
    assert v.doubtful and "revisar la fuente" in v.why
    mala = json.loads(json.dumps(ss01))
    mala["fuentes_aceptadas"][0]["cita"] = "el estudiante debe tener aprobado el 90% de sus créditos"
    assert any(EXTRACTO in p and "no aparece" in p for p in ev.verify([mala], kb_docs=BASE))
    sin_extracto = BASE - {EXTRACTO}
    assert any("no está en la base" in p for p in ev.verify([ss01], kb_docs=sin_extracto))
    row = ev.review_rows([ss01])[0]
    assert "o bien Extracto Manual Lineamientos Servicio Social" in row[5] and "70%" in row[4]


def test_base_solo_manual_y_reglamento_casos_marcados_con_propuesta():
    """2026-11-05: la base quedó con el Manual y el Reglamento. Ningún caso vigente
    cita otro documento; los afectados se marcan (no se borran) con su propuesta."""
    for c in CASES:
        assert ev.estado(c) in ev.ESTADOS, c["id"]
        if c["esperado"] == ev.ANSWER and ev.estado(c) == ev.VIGENTE:
            assert c["documento"] in BASE, c["id"]
        if ev.estado(c) != ev.VIGENTE:
            assert c["propuesta"]["nota"], c["id"]
            assert ev.with_proposal(c)["documento"] in BASE, c["id"]
    citaban_retirado = {c["id"] for c in CASES if c.get("documento") and c["documento"] not in BASE}
    assert citaban_retirado == {"ss-06"}  # ss-01, ss-02 y ss-03 ya citan el Manual
    ss01 = next(c for c in CASES if c["id"] == "ss-01")  # confirmado por la coordinadora
    assert ss01["pagina"] == 183 and "70%" in ss01["cita"] and "coordinadora" in ss01["nota"]
    assert {c["id"] for c in CASES if ev.estado(c) != ev.VIGENTE} == citaban_retirado
    assert all(ev.estado(c) == "respuesta_cambia" for c in CASES if c["id"] in citaban_retirado)
    assert ev.verify(CASES, kb_docs=BASE) == []


def test_verify_detecta_documento_retirado_sin_marcar_y_revisa_la_propuesta():
    ss = next(c for c in CASES if c["id"] == "ss-06")
    sin_marca = {k: v for k, v in ss.items() if k not in ("estado", "propuesta")}
    assert any("no está en la base de conocimiento" in p
               for p in ev.verify([sin_marca], kb_docs=BASE))
    mala = json.loads(json.dumps(ss))
    mala["propuesta"]["cita"] = "El servicio social dura tres años y se paga con beca."
    assert any("no aparece" in p for p in ev.verify([mala], kb_docs=BASE))
    sin_nota = json.loads(json.dumps(ss))
    sin_nota["propuesta"].pop("nota")
    assert any("le falta su propuesta" in p for p in ev.verify([sin_nota], kb_docs=BASE))
    assert any("estado desconocido" in p
               for p in ev.verify([{**ss, "estado": "quien sabe"}], kb_docs=BASE))


def test_kb_documents_lee_el_manifiesto(tmp_path):
    assert ev.kb_documents(tmp_path) is None
    (tmp_path / "manifest.json").write_text(json.dumps(
        [{"name": "Manual_de_Lineamientos_TecNM.pdf"},
         {"name": "Viejo.pdf", "active": False}]), encoding="utf-8")
    assert ev.kb_documents(tmp_path) == {"Manual_de_Lineamientos_TecNM.pdf"}


def test_correr_omite_los_marcados_para_corregir(monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(ev, "run", lambda cases, *a: seen.setdefault("ids", [c["id"] for c in cases]) and 0)
    assert ev.main(["--correr", "--si-gastar", "--out", "x.jsonl"]) == 0
    omitted = {c["id"] for c in CASES if ev.estado(c) in ev.NO_CORRER}
    assert omitted == {"ss-06"}  # ss-01 ya se corre: lo confirmó la coordinadora
    assert omitted.isdisjoint(seen["ids"]) and len(seen["ids"]) == len(CASES) - len(omitted)
    assert "Se omite 1 caso marcado para corregir: ss-06" in capsys.readouterr().out


def test_citas_adicionales_se_verifican_y_salen_en_la_hoja():
    ss = next(c for c in CASES if c["id"] == "ss-02")
    assert [x["pagina"] for x in ss["citas_adicionales"]] == [226, 231]
    mala = json.loads(json.dumps(ss))
    mala["citas_adicionales"][1]["cita"] = "cubriendo un mínimo total de 999 horas"
    assert any("pág. 231" in p and "no aparece" in p for p in ev.verify([mala], kb_docs=BASE))
    row = ev.review_rows([ss])[0]
    assert "(pág. 226)" in row[4] and "(pág. 231)" in row[4] and "pág. 77, 226, 231" in row[5]
    marcado = ev.review_rows([next(c for c in CASES if c["id"] == "ss-06")])[0]
    assert marcado[3].startswith("PARA VALIDAR") and "Antes decía" in marcado[3]
    assert "pág. 80" in marcado[5]


# ---- clasificación --------------------------------------------------------------------------
BAJA = next(c for c in CASES if c["id"] == "baja-01")
DATOS = next(c for c in CASES if c["id"] == "der-01")
GOOD = ("Tienes 20 días hábiles (no naturales) desde el inicio oficial de los cursos, por escrito "
        "a la División de Estudios Profesionales, si ya cursaste al menos un semestre. "
        "(Manual de Lineamientos TecNM, pág. 46)")


@pytest.mark.parametrize("case,outcome,reason,text,category,doubtful", [
    (BAJA, "answered", "none", GOOD, "acierto", False),
    (BAJA, "answered", "none", GOOD.replace(" (Manual de Lineamientos TecNM, pág. 46)", ""),
     "acierto", True),                                                       # sin fuente: revisar
    (BAJA, "answered", "none", GOOD.replace("al menos un semestre", "inscrito"), "incompleta", True),
    (BAJA, "answered", "none", GOOD + " El plazo es de 20 días naturales.", "error", True),
    (BAJA, "derived", "not_in_context", "No encontré información", "derivo_de_mas", True),
    (BAJA, "derived", "high_demand", "Alta demanda", "inconcluso", True),
    (DATOS, "derived", "out_of_scope", "Acude a Servicios Escolares", "derivacion_correcta", False),
    (DATOS, "derived", "low_similarity", "Acude a la Coordinación", "derivacion_correcta", True),
    (DATOS, "answered", "none", "Sacaste 85 en cálculo", "error", True),
])
def test_clasificacion(case, outcome, reason, text, category, doubtful):
    v = ev.classify(case, outcome, reason, text)
    assert (v.category, v.doubtful) == (category, doubtful), v.why


def test_resumen_porcentajes():
    rows = ([{"id": f"a{i}", "esperado": "responder", "categoria": "acierto", "dudoso": False,
              "tokens": 1000} for i in range(6)]
            + [{"id": "a6", "esperado": "responder", "categoria": "incompleta", "dudoso": True},
               {"id": "a7", "esperado": "responder", "categoria": "error", "dudoso": True},
               {"id": "d0", "esperado": "derivar", "categoria": "derivacion_correcta", "dudoso": False},
               {"id": "d1", "esperado": "derivar", "categoria": "error", "dudoso": True},
               {"id": "x", "esperado": "responder", "categoria": "inconcluso", "dudoso": True}])
    s = ev.summarize(rows)
    assert s["evaluados"] == 10 and s["inconclusos"] == 1
    assert s["acierto_pct"] == 75.0           # 6 de 8 a responder
    assert s["derivacion_correcta_pct"] == 50.0
    assert s["error_pct"] == 20.0             # 2 de 10
    assert s["dudosos"] == ["a6", "a7", "d1", "x"] and s["tokens"] == 6000


# ---- estimación de tokens ---------------------------------------------------------------------
def test_ajuste_lineal_de_tokens():
    a, b = ev.fit_tokens([(1000, 500), (2000, 750), (3000, 1000)])
    assert a == pytest.approx(0.25) and b == pytest.approx(250)


def test_estimacion_calibrada_con_historial(monkeypatch):
    chars = {"h1": 4000, "h2": 8000, "h3": 6000}
    monkeypatch.setattr(ev, "_prompt_chars", lambda pipe, q: chars.get(q, 0))
    history = [{"q": "h1", "llamadas_llm": 1, "tokens": 1200},
               {"q": "h2", "llamadas_llm": 1, "tokens": 2200},
               {"q": "h3", "llamadas_llm": 2, "tokens": 3400}]
    cases = [{"id": "c1", "q": "h3", "esperado": "responder"},
             {"id": "c2", "q": "bloqueada", "esperado": "derivar"}]  # no llega al modelo
    est = ev.estimate(cases, None, history)
    cal = est["calibracion"]
    assert cal["tokens_por_caracter"] == 0.25 and cal["tokens_fijos"] == 200
    assert est["llegan_al_modelo"] == 1 and est["tokens_minimos"] == 1700
    # factor: 6800 medidos / (1200 + 2200 + 1700) previstos para una llamada
    assert cal["factor_regeneraciones"] == pytest.approx(6800 / 5100, abs=0.001)
    assert est["tokens_esperados"] == round(1700 * 6800 / 5100)
    assert est["tokens_peor_caso"] > est["tokens_esperados"]


# ---- corrida (sin modelo real) -------------------------------------------------------------------
def test_correr_sin_confirmacion_no_hace_nada(monkeypatch, capsys):
    monkeypatch.setattr("app.grounding_eval.build_pipeline",
                        lambda *a: pytest.fail("no debe armar el pipeline real"))
    assert ev.main(["--correr"]) == 2
    assert "gasta cuota" in capsys.readouterr().out


class FakePipe:
    """Responde con la respuesta esperada (acierto) y cuenta tokens como el medidor real."""

    def __init__(self, meter, by_q):
        self.meter, self.by_q, self.asked = meter, by_q, []

    def answer(self, q):
        self.asked.append(q)
        self.meter["calls"] += 1
        self.meter["tokens"] += 2000
        c = self.by_q[q]
        if c["esperado"] == ev.DERIVE:
            return SimpleNamespace(outcome=SimpleNamespace(value="derived"),
                                   reason=SimpleNamespace(value=c["motivo_esperado"][0]),
                                   text="Acude a la Coordinación.", hits=[], provider="", model="")
        text = f"{c['respuesta_esperada']} ({ev._doc_name(c['documento'])}, pág. {c['pagina']})"
        return SimpleNamespace(outcome=SimpleNamespace(value="answered"),
                               reason=SimpleNamespace(value="none"), text=text, hits=[],
                               provider="fake", model="fake")


def test_corrida_simulada_se_reanuda_y_respeta_el_tope(tmp_path, monkeypatch):
    cases = CASES[:4] + [c for c in CASES if c["esperado"] == ev.DERIVE][:1]
    meter = {"calls": 0, "tokens": 0, "daily_limit": "", "last_429": ""}
    pipe = FakePipe(meter, {c["q"]: c for c in cases})
    monkeypatch.setattr("app.grounding_eval.build_pipeline", lambda env, kb: (pipe, meter))
    monkeypatch.setattr(ev.time, "sleep", lambda s: None)
    out = tmp_path / "corrida.jsonl"
    assert ev.run(cases, tmp_path / ".env", tmp_path / "kb", out, token_cap=5000) == 4
    assert len(pipe.asked) == 3  # 2000 + 2000 + 2000 >= 5000: se detiene antes del 4o
    assert ev.run(cases, tmp_path / ".env", tmp_path / "kb", out) == 0
    rows = [json.loads(l) for l in out.read_text("utf-8").splitlines()]
    assert [r["id"] for r in rows] == [c["id"] for c in cases]  # sin repetir los ya hechos
    assert {r["categoria"] for r in rows} == {"acierto", "derivacion_correcta"}
    text = ev.report(rows, cases, tmp_path / "revision.csv")
    assert "Acierto (de los que debía responder):            100.0%" in text
    with (tmp_path / "revision.csv").open(encoding="utf-8-sig") as f:
        assert len(list(csv.reader(f))) == len(cases) + 1


def test_corrida_se_detiene_con_el_tope_diario_de_groq(tmp_path, monkeypatch):
    meter = {"calls": 0, "tokens": 0, "daily_limit": "", "last_429": ""}
    pipe = FakePipe(meter, {c["q"]: c for c in CASES})
    original = pipe.answer

    def answer(q):
        meter["daily_limit"] = "Rate limit reached ... tokens per day (TPD)"
        return original(q)

    pipe.answer = answer
    monkeypatch.setattr("app.grounding_eval.build_pipeline", lambda env, kb: (pipe, meter))
    monkeypatch.setattr(ev.time, "sleep", lambda s: None)
    out = tmp_path / "corrida.jsonl"
    assert ev.run(CASES, tmp_path / ".env", tmp_path / "kb", out, token_cap=0) == 3
    assert not out.exists() or out.read_text("utf-8") == ""


# ---- archivo para la coordinadora ---------------------------------------------------------------
def test_archivo_para_la_coordinadora(tmp_path):
    from docx import Document

    docx_path, csv_path = ev.write_review(CASES, tmp_path / "revision.docx")
    table = Document(docx_path).tables[0]
    assert len(table.rows) == len(CASES) + 1
    header = [c.text for c in table.rows[0].cells]
    assert header == ev.REVIEW_HEADER and "¿Correcta o corregir?" in header
    first = [c.text for c in table.rows[1].cells]
    assert CASES[0]["q"] in first[2] and "pág." in first[5] and "Correcta" in first[6]
    with csv_path.open(encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ev.REVIEW_HEADER and len(rows) == len(CASES) + 1
    assert all(r[6] == "" for r in rows[1:])  # la coordinadora la llena
    assert any(r[3].startswith("DEBE DERIVAR") for r in rows[1:])
