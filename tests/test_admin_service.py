from app.adapters.db_sqlite import SqliteRepository
from app.admin_service import AdminService


def svc(kb, cfg):
    return AdminService(cfg, kb, SqliteRepository(cfg.db_path, cfg.session_timeout_minutes))


def test_list_documents_reflects_already_ingested_kb(kb, cfg):
    s = svc(kb, cfg)
    names = {d.name for d in s.list_documents()}
    assert names == {"Reglamento Escolar.txt", "Residencia.txt"}


def test_upload_new_document_indexes_and_syncs_to_db(kb, cfg):
    s = svc(kb, cfg)
    content = ("Requisitos de titulación: haber acreditado el 100% de créditos y el "
              "servicio social. " * 10).encode("utf-8")
    report = s.upload_document("Titulacion.txt", content, version="2026-1")
    assert report.status == "added" and report.n_chunks > 0

    names = {d.name for d in s.list_documents()}
    assert "Titulacion.txt" in names
    row = s.repo._conn.execute(
        "SELECT * FROM documentos WHERE nombre_archivo='Titulacion.txt'").fetchone()
    assert row is not None and row["version"] == "2026-1"


def test_reupload_same_name_replaces_not_duplicates(kb, cfg):
    s = svc(kb, cfg)
    content1 = ("Texto original sobre bajas temporales. " * 10).encode("utf-8")
    content2 = ("Texto actualizado sobre bajas temporales con más detalle. " * 10).encode("utf-8")
    r1 = s.upload_document("Bajas.txt", content1, "v1")
    r2 = s.upload_document("Bajas.txt", content2, "v2")
    assert r1.status == "added" and r2.status == "replaced"
    docs = [d for d in s.list_documents() if d.name == "Bajas.txt"]
    assert len(docs) == 1 and docs[0].version == "v2"


def test_unsupported_extension_is_rejected_without_touching_kb(kb, cfg):
    s = svc(kb, cfg)
    before = len(s.list_documents())
    report = s.upload_document("virus.exe", b"contenido cualquiera")
    assert report.status == "error" and "no soportado" in report.message
    assert len(s.list_documents()) == before


def test_empty_file_is_rejected(kb, cfg):
    s = svc(kb, cfg)
    report = s.upload_document("vacio.txt", b"   ")
    assert report.status == "error" and "vacío" in report.message


def test_oversized_file_is_rejected(kb, cfg):
    from app.admin_service import MAX_UPLOAD_BYTES
    s = svc(kb, cfg)
    report = s.upload_document("enorme.txt", b"x" * (MAX_UPLOAD_BYTES + 1))
    assert report.status == "error" and "excede" in report.message


def test_path_traversal_in_filename_is_neutralized(kb, cfg, tmp_path):
    s = svc(kb, cfg)
    report = s.upload_document("../../etc/evil.txt", b"contenido de prueba " * 5)
    assert report.status == "added"
    # el archivo debe quedar DENTRO de docs_dir, nunca fuera
    saved = list(cfg.docs_dir.glob("evil.txt"))
    assert len(saved) == 1
    assert not (cfg.docs_dir.parent.parent / "etc" / "evil.txt").exists()


def test_metrics_summary_delegates_to_repo(kb, cfg):
    s = svc(kb, cfg)
    m = s.metrics_summary()
    assert m.total == 0  # repo nuevo, sin interacciones todavía


def test_sanitize_filename_replaces_windows_forbidden_characters():
    from app.admin_service import sanitize_filename
    assert sanitize_filename("<img src=x>.txt") == "_img src=x_.txt"
    assert sanitize_filename("a:b|c?d*e.txt") == "a_b_c_d_e.txt"
    assert sanitize_filename('comillas"raras".pdf') == "comillas_raras_.pdf"


def test_sanitize_filename_strips_trailing_dots_and_spaces():
    from app.admin_service import sanitize_filename
    assert sanitize_filename("  nombre con espacios.txt  ") == "nombre con espacios.txt"


def test_sanitize_filename_avoids_windows_reserved_device_names():
    from app.admin_service import sanitize_filename
    assert sanitize_filename("CON.txt") == "CON_.txt"
    assert sanitize_filename("nul.pdf") == "nul_.pdf"
    assert sanitize_filename("control.txt") == "control.txt"  # no es un nombre reservado


def test_sanitize_filename_never_returns_empty():
    from app.admin_service import sanitize_filename
    assert sanitize_filename("////") == "documento"
    assert sanitize_filename("...") == "documento"


def test_sanitize_filename_leaves_normal_names_unchanged():
    from app.admin_service import sanitize_filename
    assert sanitize_filename("Reglamento Escolar 2026.pdf") == "Reglamento Escolar 2026.pdf"


def test_upload_with_windows_forbidden_characters_does_not_crash(kb, cfg):
    s = svc(kb, cfg)
    nombre = "<img src=x onerror=alert(1)>.txt"
    report = s.upload_document(nombre, b"contenido inofensivo de prueba " * 5)
    assert report.status == "added"
    assert "<" not in report.name and ">" not in report.name
    saved = [p for p in cfg.docs_dir.iterdir() if "img src" in p.name]
    assert len(saved) == 1 and "<" not in saved[0].name


def test_panel_ve_la_reindexacion_hecha_desde_la_cli_y_no_la_revierte(kb, cfg, embedder, tmp_path):
    # 2026-11-05: se reindexó solo con 2 documentos desde la CLI mientras el panel
    # seguía abierto. El panel listaba los viejos y, al subir uno, habría guardado
    # su copia en memoria encima del índice nuevo (los retirados regresaban).
    import os
    from app.rag.knowledge_base import KnowledgeBase
    s = svc(kb, cfg)
    assert {d.name for d in s.list_documents()} == {"Reglamento Escolar.txt", "Residencia.txt"}
    nuevo = KnowledgeBase.open(cfg, embedder, rebuild=True)
    nuevo.ingest_file(cfg.docs_dir / "Reglamento Escolar.txt")
    nuevo.save()
    mp = cfg.kb_dir / "manifest.json"
    st = mp.stat()
    os.utime(mp, (st.st_atime + 5, st.st_mtime + 5))
    assert {d.name for d in s.list_documents()} == {"Reglamento Escolar.txt"}
    otra = KnowledgeBase.open(cfg, embedder, rebuild=True)  # otra vez desde la CLI,
    otra.ingest_file(cfg.docs_dir / "Residencia.txt")       # ahora sin listar antes de subir
    otra.save()
    os.utime(mp, (st.st_atime + 10, st.st_mtime + 10))
    content = ("Requisitos de titulación: haber acreditado el 100% de créditos. " * 10).encode()
    assert s.upload_document("Titulacion.txt", content).status == "added"
    disco = KnowledgeBase.open(cfg, embedder)
    assert {d.name for d in disco.docs.values()} == {"Residencia.txt", "Titulacion.txt"}
