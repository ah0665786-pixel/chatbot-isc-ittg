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
