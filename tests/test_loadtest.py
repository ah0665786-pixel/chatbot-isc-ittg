from app.loadtest import FakeFastLLM, SAMPLE_QUESTIONS, _sandbox_service, run


def test_fake_llm_adds_realistic_latency():
    llm = FakeFastLLM(min_s=0.05, max_s=0.08)
    t0 = __import__("time").perf_counter()
    reply = llm.chat([{"role": "user", "content": "x"}])
    elapsed = __import__("time").perf_counter() - t0
    assert 0.05 <= elapsed <= 0.3 and reply.provider == "fake"


def test_concurrent_run_produces_all_results_without_errors(cfg):
    n_users, msgs = 12, 2
    results, wall = run(cfg, n_users=n_users, messages_per_user=msgs,
                        llm_min_s=0.05, llm_max_s=0.1)
    assert len(results) == n_users * msgs
    assert all(r.ok for r in results)  # cero errores: RNF-EF-02
    assert {r.user for r in results} == {f"carga-{i:03d}" for i in range(n_users)}


def test_requests_actually_run_in_parallel_not_serialized(cfg):
    # 20 usuarios x ~0.2s cada uno: en serie tomarían ~4s; en paralelo, mucho
    # menos. Si esto falla, significa que algo (probablemente el candado de
    # la base de datos) está serializando las peticiones sin querer.
    n_users = 20
    results, wall = run(cfg, n_users=n_users, messages_per_user=1,
                        llm_min_s=0.2, llm_max_s=0.2)
    serial_estimate = n_users * 0.2
    assert wall < serial_estimate * 0.5, (
        f"wall={wall:.2f}s no es mucho menor que el estimado en serie "
        f"({serial_estimate:.2f}s) — ¿se perdió la concurrencia?")


def test_database_integrity_after_concurrent_writes(cfg):
    n_users, msgs = 15, 3
    results, wall = run(cfg, n_users=n_users, messages_per_user=msgs,
                        llm_min_s=0.02, llm_max_s=0.05)
    from app.adapters.db_sqlite import SqliteRepository
    from dataclasses import replace
    from pathlib import Path
    test_cfg = replace(cfg, db_path=Path("data/loadtest.db"))
    repo = SqliteRepository(test_cfg.db_path, test_cfg.session_timeout_minutes)
    try:
        n_conv = repo._conn.execute("SELECT COUNT(*) n FROM conversaciones").fetchone()["n"]
        n_msg = repo._conn.execute("SELECT COUNT(*) n FROM mensajes").fetchone()["n"]
        n_met = repo._conn.execute("SELECT COUNT(*) n FROM metricas").fetchone()["n"]
        assert n_conv == n_users          # una conversación por usuario, sin duplicados
        assert n_msg == n_users * msgs * 2  # cada consulta = 1 mensaje usuario + 1 respuesta
        assert n_met == n_users * msgs      # una métrica por consulta, ninguna perdida
    finally:
        repo.close()


def test_loadtest_does_not_touch_real_project_paths(cfg, tmp_path):
    # cfg (fixture) ya usa tmp_path para kb/db "reales" del proyecto de
    # prueba; confirmamos que loadtest usa SUS PROPIAS rutas, no esas.
    run(cfg, n_users=3, messages_per_user=1, llm_min_s=0.01, llm_max_s=0.02)
    assert not (cfg.db_path).exists() or cfg.db_path == Path("data/loadtest.db")
    from pathlib import Path
    assert not Path(cfg.kb_dir, "meta.json").exists()  # el kb "real" del fixture sigue vacío


def test_all_sample_questions_actually_reach_the_llm(cfg):
    # Protege contra el bug real que encontramos: si el umbral del sandbox
    # queda demasiado alto para HashingEmbedder, las preguntas se derivan
    # localmente SIN llamar al LLM, y la prueba de carga deja de medir lo
    # que dice medir (deja de ser una prueba de concurrencia real).
    from app.domain import Outcome
    svc = _sandbox_service(cfg, 0.01, 0.02)
    try:
        for q in SAMPLE_QUESTIONS:
            result, _ = svc.handle_message("debug", "debug", q)
            assert result.outcome is Outcome.ANSWERED, (
                f"'{q}' se derivó sin llegar al LLM (outcome={result.outcome.value}, "
                f"reason={result.reason.value}) — la prueba de carga no estaría midiendo "
                f"concurrencia real")
    finally:
        svc.repo.close()
