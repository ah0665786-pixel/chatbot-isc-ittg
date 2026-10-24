from app.telegram_format import markdown_to_telegram_html, strip_markdown


def test_negrita_se_convierte_en_html():
    assert markdown_to_telegram_html("Necesitas **80% de créditos**.") == \
        "Necesitas <b>80% de créditos</b>."


def test_negrita_con_guiones_bajos():
    assert markdown_to_telegram_html("__Importante__: entrega") == "<b>Importante</b>: entrega"


def test_escapa_caracteres_html():
    # "<" sin escapar haría que Telegram rechazara el mensaje completo
    assert markdown_to_telegram_html("promedio < 70 & **baja**") == \
        "promedio &lt; 70 &amp; <b>baja</b>"


def test_titulos_y_vinetas():
    texto = "### Requisitos\n* Constancia\n- Kárdex"
    assert markdown_to_telegram_html(texto) == "<b>Requisitos</b>\n• Constancia\n• Kárdex"


def test_codigo_no_se_toca():
    assert markdown_to_telegram_html("usa `**no**` aquí") == "usa <code>**no**</code> aquí"


def test_asteriscos_sin_pareja_se_quitan():
    assert markdown_to_telegram_html("texto **suelto") == "texto suelto"


def test_texto_sin_formato_queda_igual():
    assert markdown_to_telegram_html("Hola, ¿en qué te ayudo?") == "Hola, ¿en qué te ayudo?"


def test_strip_markdown_quita_las_marcas():
    assert strip_markdown("### Paso\n**Ve** a `servicios`") == "Paso\nVe a servicios"


# ---- división de respuestas largas (HU-06 CA-03) -------------------------
from app.telegram_format import SAFE_LIMIT, TELEGRAM_LIMIT, split_markdown  # noqa: E402


def _balanced(html_text: str) -> bool:
    return all(html_text.count(f"<{t}>") == html_text.count(f"</{t}>")
               for t in ("b", "code", "pre"))


def test_texto_corto_queda_en_un_solo_trozo():
    assert split_markdown("**Hola** mundo") == ["**Hola** mundo"]


def test_lista_larga_se_corta_por_parrafo_y_cada_trozo_cabe():
    paras = [f"**Opción {i}**: " + "requisito del reglamento " * 12 for i in range(40)]
    text = "\n\n".join(p.strip() for p in paras)
    parts = split_markdown(text)
    assert len(parts) > 1
    for p in parts:
        h = markdown_to_telegram_html(p)
        assert len(h) <= SAFE_LIMIT < TELEGRAM_LIMIT and _balanced(h)
    assert "\n\n".join(parts) == text  # no se pierde ni se parte ningún párrafo


def test_parrafo_gigante_se_corta_por_linea_y_luego_por_palabra():
    lines = "\n".join(f"- punto {i} con **negrita** y texto" for i in range(300))
    one_line = "palabra " * 1500
    for text in (lines, one_line):
        parts = split_markdown(text)
        assert len(parts) > 1
        for p in parts:
            h = markdown_to_telegram_html(p)
            assert len(h) <= SAFE_LIMIT and _balanced(h)
        assert " ".join(" ".join(parts).split()) == " ".join(text.split())


def test_escape_html_cuenta_para_el_limite():
    # "&" se vuelve "&amp;" (5 caracteres): el Markdown cabe, el HTML no
    text = "&" * 1000
    parts = split_markdown(text)
    assert len(parts) == 2 and "".join(parts) == text
    assert all(len(markdown_to_telegram_html(p)) <= SAFE_LIMIT for p in parts)


def test_bloque_de_codigo_no_se_separa_entre_parrafos():
    code = "```\nlinea 1\n\nlinea 2\n```"
    text = ("a " * 1500).strip() + "\n\n" + code + "\n\n" + ("b " * 1500).strip()
    assert any(code in p for p in split_markdown(text))
