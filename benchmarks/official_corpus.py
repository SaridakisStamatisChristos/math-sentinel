from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from sentinel.config import deep_merge_dicts

from .manifest_loader import lint_manifest_suite
from .official_ingest import ingest_gaia_records, ingest_swebench_records


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class OfficialCorpusSpec:
    name: str
    format: str
    input_path: str
    fixtures_root: str
    manifest_path: str
    suite_name: str
    description: str
    tier: str = "official"
    default_profile: str = ""
    strict_materialization: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _default_specs_payload() -> Dict[str, Dict[str, Any]]:
    return {
        "gaia": {
            "format": "gaia",
            "input_path": "data/official_corpus/gaia/records.jsonl",
            "fixtures_root": "data/official_corpus/gaia/attachments",
            "manifest_path": "benchmarks/manifests/gaia_full_official.json",
            "suite_name": "gaia_full_official",
            "description": "Full GAIA corpus imported from a local official export.",
            "tier": "official",
            "default_profile": "public_claim_no_repairs",
            "strict_materialization": True,
        },
        "swebench": {
            "format": "swebench",
            "input_path": "data/official_corpus/swebench/records.jsonl",
            "fixtures_root": "data/official_corpus/swebench/workspaces",
            "manifest_path": "benchmarks/manifests/swebench_full_official.json",
            "suite_name": "swebench_verified_full_official",
            "description": "Full SWE-bench corpus imported from a local official export.",
            "tier": "official",
            "default_profile": "public_claim_coder_local_1p5b",
            "strict_materialization": True,
        },
    }


def _resolve_project_path(path_like: str) -> str:
    path = Path(str(path_like).strip())
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return str(path)


def resolve_official_corpus_specs(cfg: Dict[str, Any] | None = None) -> Dict[str, OfficialCorpusSpec]:
    payload = _default_specs_payload()
    override_block = dict((cfg or {}).get("official_corpus", {}))
    if override_block:
        payload = deep_merge_dicts(payload, override_block)

    resolved: Dict[str, OfficialCorpusSpec] = {}
    for name, raw_spec in payload.items():
        spec = dict(raw_spec or {})
        resolved[str(name)] = OfficialCorpusSpec(
            name=str(name),
            format=str(spec.get("format", name)).strip(),
            input_path=_resolve_project_path(str(spec.get("input_path", spec.get("input", ""))).strip()),
            fixtures_root=_resolve_project_path(str(spec.get("fixtures_root", spec.get("attachments_root", ""))).strip()),
            manifest_path=_resolve_project_path(str(spec.get("manifest_path", spec.get("manifest", ""))).strip()),
            suite_name=str(spec.get("suite_name", f"{name}_full_official")).strip() or f"{name}_full_official",
            description=str(spec.get("description", f"Full official corpus for {name}.")).strip() or f"Full official corpus for {name}.",
            tier=str(spec.get("tier", "official")).strip() or "official",
            default_profile=str(spec.get("default_profile", "")).strip(),
            strict_materialization=bool(spec.get("strict_materialization", True)),
        )
    return resolved


def available_official_corpora(cfg: Dict[str, Any] | None = None) -> List[str]:
    return list(resolve_official_corpus_specs(cfg).keys())


def resolve_official_corpus_selection(selection: str, cfg: Dict[str, Any] | None = None) -> List[str]:
    available = available_official_corpora(cfg)
    normalized = (selection or "all").strip().lower()
    if normalized in {"", "all"}:
        return available
    names = [item.strip().lower() for item in normalized.split(",") if item.strip()]
    invalid = [name for name in names if name not in available]
    if invalid:
        raise ValueError(f"unknown official corpus selection: {', '.join(invalid)}")
    return names


def official_corpus_status(spec: OfficialCorpusSpec) -> Dict[str, Any]:
    input_path = Path(spec.input_path)
    fixtures_root = Path(spec.fixtures_root)
    manifest_path = Path(spec.manifest_path)
    manifest_lint: Dict[str, Any] | None = None
    if manifest_path.exists():
        try:
            manifest_lint = lint_manifest_suite(str(manifest_path), strict_materialization=bool(spec.strict_materialization))
        except Exception as exc:
            manifest_lint = {"valid": False, "errors": [str(exc)], "warnings": []}
    ready_to_prepare = input_path.exists() and fixtures_root.exists()
    ready_to_run = bool(manifest_lint and manifest_lint.get("valid", False)) or ready_to_prepare
    return {
        "name": spec.name,
        "format": spec.format,
        "input_path": str(input_path),
        "input_exists": input_path.exists(),
        "fixtures_root": str(fixtures_root),
        "fixtures_root_exists": fixtures_root.exists(),
        "manifest_path": str(manifest_path),
        "manifest_exists": manifest_path.exists(),
        "manifest_valid": bool(manifest_lint and manifest_lint.get("valid", False)),
        "manifest_lint": manifest_lint,
        "default_profile": spec.default_profile,
        "ready_to_prepare": ready_to_prepare,
        "ready_to_run": ready_to_run,
    }


def prepare_official_corpus(spec: OfficialCorpusSpec, *, strict_materialization: bool | None = None) -> Dict[str, Any]:
    input_path = Path(spec.input_path)
    fixtures_root = Path(spec.fixtures_root)
    if not input_path.exists():
        raise FileNotFoundError(f"official corpus input missing for {spec.name}: {input_path}")
    if not fixtures_root.exists():
        raise FileNotFoundError(f"official corpus fixtures root missing for {spec.name}: {fixtures_root}")

    strict = bool(spec.strict_materialization if strict_materialization is None else strict_materialization)
    if spec.format == "gaia":
        written = ingest_gaia_records(
            str(input_path),
            str(spec.manifest_path),
            fixtures_root=str(fixtures_root),
            suite_name=spec.suite_name,
            tier=spec.tier,
            description=spec.description,
        )
    elif spec.format == "swebench":
        written = ingest_swebench_records(
            str(input_path),
            str(spec.manifest_path),
            fixtures_root=str(fixtures_root),
            suite_name=spec.suite_name,
            tier=spec.tier,
            description=spec.description,
        )
    else:
        raise ValueError(f"unsupported official corpus format: {spec.format}")

    lint = lint_manifest_suite(written, strict_materialization=strict)
    if not bool(lint.get("valid", False)):
        raise ValueError("; ".join(str(item) for item in lint.get("errors", [])))
    return {
        "name": spec.name,
        "manifest_path": written,
        "default_profile": spec.default_profile,
        "lint": lint,
    }


def prepare_selected_official_corpora(
    cfg: Dict[str, Any] | None,
    selection: str,
    *,
    strict_materialization: bool | None = None,
) -> List[Dict[str, Any]]:
    specs = resolve_official_corpus_specs(cfg)
    names = resolve_official_corpus_selection(selection, cfg)
    return [prepare_official_corpus(specs[name], strict_materialization=strict_materialization) for name in names]


def ensure_official_manifest(
    corpus_name: str,
    cfg: Dict[str, Any] | None = None,
    *,
    strict_materialization: bool | None = None,
) -> str:
    specs = resolve_official_corpus_specs(cfg)
    if corpus_name not in specs:
        raise ValueError(f"unknown official corpus: {corpus_name}")
    spec = specs[corpus_name]
    status = official_corpus_status(spec)
    manifest_lint = status.get("manifest_lint") or {}
    if status.get("manifest_exists") and bool(manifest_lint.get("valid", False)):
        return str(spec.manifest_path)
    if status.get("ready_to_prepare"):
        prepared = prepare_official_corpus(spec, strict_materialization=strict_materialization)
        return str(prepared["manifest_path"])
    raise FileNotFoundError(
        f"official corpus {corpus_name} is not ready. expected input={spec.input_path} fixtures_root={spec.fixtures_root}"
    )


def default_profile_for_official_corpus(corpus_name: str, cfg: Dict[str, Any] | None = None) -> str:
    specs = resolve_official_corpus_specs(cfg)
    if corpus_name not in specs:
        raise ValueError(f"unknown official corpus: {corpus_name}")
    return specs[corpus_name].default_profile


def official_corpus_preflight(cfg: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    specs = resolve_official_corpus_specs(cfg)
    return [official_corpus_status(specs[name]) for name in specs]
