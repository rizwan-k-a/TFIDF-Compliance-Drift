"""Repository Audit: dependency graph, call graph, dead-code, duplication, and loop inefficiency.

This is a best-effort static analysis for this repo (Python is dynamic).
It is designed to be deterministic, low-dependency, and runnable in CI.

Usage (from repo root):
  python scripts/audit_repo.py --out results/audit_report.md

It also writes machine-readable artifacts next to the report:
  - import_graph.dot
  - call_graph.dot
  - audit.json
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FileInfo:
    path: Path
    module_id: str
    kind: str  # src|dashboard|scripts|tests|root


@dataclass(frozen=True)
class FuncInfo:
    module_id: str
    qualname: str
    path: Path
    lineno: int

    @property
    def node_id(self) -> str:
        return f"{self.module_id}:{self.qualname}"


def iter_py_files() -> List[Path]:
    files: List[Path] = []
    for p in REPO_ROOT.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        files.append(p)
    return sorted(files)


def classify_file(path: Path) -> FileInfo:
    rel = path.relative_to(REPO_ROOT)
    parts = rel.parts

    if parts[:1] == ("src",):
        return FileInfo(path=path, module_id=rel.stem, kind="src")
    if parts[:1] == ("dashboard",):
        return FileInfo(path=path, module_id=f"dashboard.{rel.stem}", kind="dashboard")
    if parts[:1] == ("scripts",):
        return FileInfo(path=path, module_id=f"scripts.{rel.stem}", kind="scripts")
    if parts[:1] == ("tests",):
        return FileInfo(path=path, module_id=f"tests.{rel.stem}", kind="tests")

    return FileInfo(path=path, module_id=rel.stem, kind="root")


def safe_parse(path: Path) -> Tuple[Optional[ast.Module], Optional[str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")

    try:
        return ast.parse(text, filename=str(path)), None
    except SyntaxError as e:
        return None, f"SyntaxError: {e}"


def extract_imports(tree: ast.Module) -> Tuple[Set[str], Set[Tuple[str, str]]]:
    """Return (imported_modules, from_imported_symbols).

    - imported_modules: names like 'vectorize', 'sklearn', 'dashboard.app'
    - from_imported_symbols: pairs like ('manual_tfidf_math', 'tokenize')
    """

    imported: Set[str] = set()
    from_symbols: Set[Tuple[str, str]] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if not node.module:
                continue
            imported.add(node.module)
            for alias in node.names:
                if alias.name and alias.name != "*":
                    from_symbols.add((node.module, alias.name))

    return imported, from_symbols


class DefAndCallVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.func_defs: List[Tuple[str, int]] = []  # (qualname, lineno)
        self.calls: List[Tuple[str, str, int]] = []  # (caller_qualname|<module>, callee_expr, lineno)
        self.nested_loops: List[Tuple[str, int, int]] = []  # (loop_type, lineno, depth)
        self.branch_constants: List[Tuple[int, str]] = []  # (lineno, desc)
        self.redundant_checks: List[Tuple[int, str]] = []  # (lineno, test_src)

        self._func_stack: List[str] = []
        self._class_stack: List[str] = []
        self._loop_depth = 0
        self._seen_tests_in_scope: List[Counter[str]] = [Counter()]

    def _qualname(self, name: str) -> str:
        scope = self._class_stack + self._func_stack
        if not scope:
            return name
        return ".".join(scope + [name])

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        qn = self._qualname(node.name)
        self.func_defs.append((qn, node.lineno))

        self._func_stack.append(node.name)
        self._seen_tests_in_scope.append(Counter())
        self.generic_visit(node)
        self._seen_tests_in_scope.pop()
        self._func_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        # Treat same as FunctionDef
        qn = self._qualname(node.name)
        self.func_defs.append((qn, node.lineno))

        self._func_stack.append(node.name)
        self._seen_tests_in_scope.append(Counter())
        self.generic_visit(node)
        self._seen_tests_in_scope.pop()
        self._func_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        callee = self._format_callee(node.func)
        if callee:
            caller = ".".join(self._class_stack + self._func_stack) if (self._class_stack or self._func_stack) else "<module>"
            self.calls.append((caller, callee, node.lineno))
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self._loop_depth += 1
        if self._loop_depth >= 2:
            self.nested_loops.append(("for", node.lineno, self._loop_depth))
        self.generic_visit(node)
        self._loop_depth -= 1

    def visit_While(self, node: ast.While) -> None:
        self._loop_depth += 1
        if self._loop_depth >= 2:
            self.nested_loops.append(("while", node.lineno, self._loop_depth))
        self.generic_visit(node)
        self._loop_depth -= 1

    def visit_If(self, node: ast.If) -> None:
        # Unreachable-branch heuristic: constant tests
        const_val = self._const_bool(node.test)
        if const_val is False:
            self.branch_constants.append((node.lineno, "if False (body unreachable)"))
        elif const_val is True:
            self.branch_constants.append((node.lineno, "if True (else unreachable)"))

        test_src = self._stable_test_repr(node.test)
        if test_src:
            scope_counter = self._seen_tests_in_scope[-1]
            scope_counter[test_src] += 1
            if scope_counter[test_src] >= 2:
                self.redundant_checks.append((node.lineno, test_src))

        self.generic_visit(node)

    @staticmethod
    def _format_callee(func: ast.AST) -> Optional[str]:
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            # foo.bar(...) => "foo.bar" where foo is Name
            if isinstance(func.value, ast.Name):
                return f"{func.value.id}.{func.attr}"
            return func.attr
        return None

    @staticmethod
    def _const_bool(expr: ast.AST) -> Optional[bool]:
        if isinstance(expr, ast.Constant) and isinstance(expr.value, bool):
            return bool(expr.value)
        return None

    @staticmethod
    def _stable_test_repr(expr: ast.AST) -> Optional[str]:
        # A compact representation that is stable across formatting.
        try:
            dumped = ast.dump(expr, include_attributes=False)
        except Exception:
            return None
        # Trim very long expressions
        return dumped[:400]


def normalize_func_ast(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Return a normalized ast.dump to detect exact structural duplicates."""
    return ast.dump(node, include_attributes=False)


def build_index(files: List[FileInfo]) -> Tuple[Dict[str, FileInfo], Dict[str, Path]]:
    """Return:
    - module_id -> FileInfo
    - import_name -> file path (resolution table)

    Note: src modules are imported as their stem (e.g., 'vectorize').
    """

    by_module_id: Dict[str, FileInfo] = {fi.module_id: fi for fi in files}

    import_name_to_path: Dict[str, Path] = {}

    for fi in files:
        rel = fi.path.relative_to(REPO_ROOT)
        if fi.kind == "src":
            import_name_to_path[rel.stem] = fi.path
        elif fi.kind in {"dashboard", "scripts", "tests"}:
            import_name_to_path[fi.module_id] = fi.path
            # Also allow importing by leaf name (common for scripts/tests if sys.path changes)
            import_name_to_path[rel.stem] = fi.path
        else:
            import_name_to_path[fi.module_id] = fi.path
            import_name_to_path[rel.stem] = fi.path

    return by_module_id, import_name_to_path


def resolve_local_import(import_name_to_path: Dict[str, Path], imported_module: str) -> Optional[Path]:
    if imported_module in import_name_to_path:
        return import_name_to_path[imported_module]

    # Try top-level package segment
    root = imported_module.split(".")[0]
    return import_name_to_path.get(root)


def find_entrypoints(files: List[FileInfo]) -> Dict[str, List[Path]]:
    """Entry points are not only __main__ blocks; operational vs utilities matter."""

    operational: List[Path] = []
    utilities: List[Path] = []
    tests: List[Path] = []
    debug_scripts: List[Path] = []

    for fi in files:
        rel = fi.path.relative_to(REPO_ROOT).as_posix()
        if rel == "dashboard/app.py":
            operational.append(fi.path)
            continue
        if rel == "setup_validate.py":
            utilities.append(fi.path)
            continue
        if rel.startswith("scripts/"):
            utilities.append(fi.path)
            continue
        if rel.startswith("tests/"):
            tests.append(fi.path)
            continue

        # Heuristic: __main__ block in src modules indicates debug entrypoint
        if fi.kind == "src":
            tree, err = safe_parse(fi.path)
            if tree and "__main__" in fi.path.read_text(encoding="utf-8", errors="ignore"):
                debug_scripts.append(fi.path)

    return {
        "operational": sorted(set(operational)),
        "utilities": sorted(set(utilities)),
        "tests": sorted(set(tests)),
        "debug": sorted(set(debug_scripts)),
    }


def reachable_from(
    roots: Iterable[Path],
    local_import_edges: Dict[Path, Set[Path]],
) -> Set[Path]:
    seen: Set[Path] = set()
    stack = list(roots)

    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for nxt in sorted(local_import_edges.get(cur, set())):
            if nxt not in seen:
                stack.append(nxt)

    return seen


def build_import_graph(files: List[FileInfo]) -> Tuple[Dict[Path, Set[Path]], Dict[Path, Set[str]], Dict[Path, Set[Tuple[str, str]]], Dict[Path, Optional[str]]]:
    _, import_name_to_path = build_index(files)

    local_edges: Dict[Path, Set[Path]] = defaultdict(set)
    imported_modules_by_file: Dict[Path, Set[str]] = {}
    from_symbols_by_file: Dict[Path, Set[Tuple[str, str]]] = {}
    parse_errors: Dict[Path, Optional[str]] = {}

    for fi in files:
        tree, err = safe_parse(fi.path)
        parse_errors[fi.path] = err
        if not tree:
            imported_modules_by_file[fi.path] = set()
            from_symbols_by_file[fi.path] = set()
            continue

        imported, from_symbols = extract_imports(tree)
        imported_modules_by_file[fi.path] = imported
        from_symbols_by_file[fi.path] = from_symbols

        for mod in imported:
            target = resolve_local_import(import_name_to_path, mod)
            if target and target != fi.path:
                local_edges[fi.path].add(target)

    return local_edges, imported_modules_by_file, from_symbols_by_file, parse_errors


def build_call_graph(files: List[FileInfo]) -> Tuple[Dict[str, FuncInfo], Dict[str, Set[str]], Dict[str, List[Tuple[str, int]]], Dict[Path, Dict[str, List[Tuple[int, str]]]]]:
    """Return:
    - funcs_by_id: node_id -> FuncInfo
    - call_edges: caller_node_id -> set(callee_node_id or external)
    - raw_calls: module_id -> list of (callee_expr, lineno)
    - heuristics: path -> {category -> [(lineno, detail)]}
    """

    module_by_path: Dict[Path, str] = {fi.path: fi.module_id for fi in files}

    funcs_by_id: Dict[str, FuncInfo] = {}
    funcs_by_module: Dict[str, Dict[str, FuncInfo]] = defaultdict(dict)  # module -> qualname -> info
    raw_calls: Dict[str, List[Tuple[str, str, int]]] = defaultdict(list)
    name_uses: Dict[str, Set[str]] = defaultdict(set)  # module_id -> set(Name.id)
    heuristics: Dict[Path, Dict[str, List[Tuple[int, str]]]] = defaultdict(lambda: defaultdict(list))

    # First pass: collect function defs + loop/branch heuristics
    for fi in files:
        tree, err = safe_parse(fi.path)
        if not tree:
            continue

        visitor = DefAndCallVisitor()
        visitor.visit(tree)

        for qn, lineno in visitor.func_defs:
            info = FuncInfo(module_id=fi.module_id, qualname=qn, path=fi.path, lineno=lineno)
            funcs_by_id[info.node_id] = info
            funcs_by_module[fi.module_id][qn] = info

        for loop_type, lineno, depth in visitor.nested_loops:
            heuristics[fi.path]["nested_loops"].append((lineno, f"{loop_type} depth={depth}"))
        for lineno, desc in visitor.branch_constants:
            heuristics[fi.path]["constant_branches"].append((lineno, desc))
        for lineno, test_src in visitor.redundant_checks:
            heuristics[fi.path]["redundant_checks"].append((lineno, test_src))

        raw_calls[fi.module_id].extend(visitor.calls)

        # Collect name uses (supports indirect calls like calling a variable holding a function)
        for n in ast.walk(tree):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                name_uses[fi.module_id].add(n.id)

    # Second pass: resolve call edges best-effort
    call_edges: Dict[str, Set[str]] = defaultdict(set)

    # Map simple name -> function in same module (top-level only)
    top_level_funcs_by_module: Dict[str, Dict[str, FuncInfo]] = {
        mod: {qn: info for qn, info in defs.items() if "." not in qn}
        for mod, defs in funcs_by_module.items()
    }

    # Also map imported-module aliases? We keep minimal; attribute calls resolved as "mod.func" string.

    for fi in files:
        module_id = fi.module_id

        for caller_qn, callee_expr, _lineno in raw_calls.get(module_id, []):
            caller_node = f"{module_id}:{caller_qn}" if caller_qn != "<module>" else f"{module_id}:<module>"
            # Local direct call: foo()
            if "." not in callee_expr:
                if callee_expr in top_level_funcs_by_module.get(module_id, {}):
                    call_edges[caller_node].add(f"{module_id}:{callee_expr}")
                else:
                    call_edges[caller_node].add(f"external:{callee_expr}")
                continue

            # Attribute call: mod.func()
            mod_part, func_part = callee_expr.split(".", 1)

            # If mod_part matches a local src module name, resolve to that module's function.
            if mod_part in top_level_funcs_by_module:
                if func_part in top_level_funcs_by_module[mod_part]:
                    call_edges[caller_node].add(f"{mod_part}:{func_part}")
                else:
                    call_edges[caller_node].add(f"external:{callee_expr}")
                continue

            call_edges[caller_node].add(f"external:{callee_expr}")

    # Store name_uses on the function for later dead-code heuristics
    build_call_graph.name_uses = name_uses  # type: ignore[attr-defined]

    return funcs_by_id, call_edges, raw_calls, heuristics


def detect_duplicate_functions(files: List[FileInfo]) -> List[Dict[str, object]]:
    """Detect exact structural duplicates across modules using AST."""

    dup_map: Dict[str, List[FuncInfo]] = defaultdict(list)

    for fi in files:
        tree, err = safe_parse(fi.path)
        if not tree:
            continue

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                norm = normalize_func_ast(node)
                # reduce noise: ignore pure docstring-only functions
                if len(node.body) == 1 and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], "value", None), ast.Constant):
                    continue
                key = f"{fi.kind}:{node.name}:{hash(norm)}"
                dup_map[key].append(FuncInfo(module_id=fi.module_id, qualname=node.name, path=fi.path, lineno=node.lineno))

    dups: List[Dict[str, object]] = []
    for key, items in dup_map.items():
        if len(items) <= 1:
            continue
        # Verify name + structure duplicates but across different files
        paths = {it.path for it in items}
        if len(paths) <= 1:
            continue
        dups.append({
            "signature": key,
            "count": len(items),
            "occurrences": [
                {"node_id": it.node_id, "path": str(it.path.relative_to(REPO_ROOT)).replace("\\", "/"), "lineno": it.lineno}
                for it in sorted(items, key=lambda x: (str(x.path), x.lineno))
            ],
        })

    return sorted(dups, key=lambda d: (-int(d["count"]), str(d["signature"])))


def detect_unnecessary_iterations(files: List[FileInfo]) -> List[Dict[str, object]]:
    findings: List[Dict[str, object]] = []

    # Highly targeted heuristics for this repo: "computed but unused" patterns.
    target = REPO_ROOT / "src" / "vectorize.py"
    if target.exists():
        text = target.read_text(encoding="utf-8", errors="ignore")
        if "cleaned_docs" in text and "fit_transform(all_docs)" in text and "fit_transform(cleaned_docs)" not in text:
            findings.append({
                "path": "src/vectorize.py",
                "lineno_hint": None,
                "issue": "cleaned_docs is built but unused; TF-IDF is fit on all_docs instead",
                "impact": "wasted iteration + defensive-cleaning is ineffective",
            })

    return findings


def write_dot_import_graph(local_edges: Dict[Path, Set[Path]], out_path: Path) -> None:
    lines = ["digraph import_graph {", "  rankdir=LR;"]
    for src, dsts in sorted(local_edges.items(), key=lambda x: str(x[0])):
        src_rel = src.relative_to(REPO_ROOT).as_posix()
        for dst in sorted(dsts, key=lambda p: str(p)):
            dst_rel = dst.relative_to(REPO_ROOT).as_posix()
            lines.append(f'  "{src_rel}" -> "{dst_rel}";')
    lines.append("}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_dot_call_graph(call_edges: Dict[str, Set[str]], out_path: Path) -> None:
    lines = ["digraph call_graph {", "  rankdir=LR;"]
    for caller, callees in sorted(call_edges.items()):
        for callee in sorted(callees):
            lines.append(f'  "{caller}" -> "{callee}";')
    lines.append("}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def render_report(
    out_md: Path,
    files: List[FileInfo],
    entrypoints: Dict[str, List[Path]],
    local_import_edges: Dict[Path, Set[Path]],
    parse_errors: Dict[Path, Optional[str]],
    funcs_by_id: Dict[str, FuncInfo],
    call_edges: Dict[str, Set[str]],
    from_symbols_by_file: Dict[Path, Set[Tuple[str, str]]],
    duplicates: List[Dict[str, object]],
    nested_loop_findings: Dict[Path, Dict[str, List[Tuple[int, str]]]],
    unnecessary_iterations: List[Dict[str, object]],
) -> Dict[str, object]:
    # Reachability modes
    operational_reach = reachable_from(entrypoints["operational"], local_import_edges)
    utilities_reach = reachable_from(entrypoints["utilities"], local_import_edges)
    tests_reach = reachable_from(entrypoints["tests"], local_import_edges)

    all_paths = {fi.path for fi in files}

    never_imported_anywhere: List[str] = []
    all_import_targets = {dst for dsts in local_import_edges.values() for dst in dsts}
    for fi in files:
        if fi.path not in all_import_targets and fi.path not in (set(entrypoints["operational"]) | set(entrypoints["utilities"]) | set(entrypoints["tests"])):
            never_imported_anywhere.append(fi.path.relative_to(REPO_ROOT).as_posix())

    never_executed_operational = sorted((all_paths - operational_reach), key=lambda p: p.as_posix())
    never_executed_operational_rel = [p.relative_to(REPO_ROOT).as_posix() for p in never_executed_operational]

    never_executed_tests = sorted((all_paths - tests_reach), key=lambda p: p.as_posix())
    never_executed_tests_rel = [p.relative_to(REPO_ROOT).as_posix() for p in never_executed_tests]

    # Dead functions (best-effort): top-level functions not called, not imported by name,
    # and not referenced as first-class objects (e.g., stored in a list for indirect calling).
    imported_by_name: Set[str] = set()
    for src_path, pairs in from_symbols_by_file.items():
        for mod, sym in pairs:
            imported_by_name.add(f"{mod}:{sym}")

    called_nodes: Set[str] = set()
    for caller, callees in call_edges.items():
        for callee in callees:
            if callee.startswith("external:"):
                continue
            called_nodes.add(callee)

    kind_by_path: Dict[Path, str] = {fi.path: fi.kind for fi in files}

    name_uses: Dict[str, Set[str]] = getattr(build_call_graph, "name_uses", {})  # type: ignore[name-defined]

    dead_funcs: List[Dict[str, object]] = []
    for node_id, info in funcs_by_id.items():
        # Only consider top-level defs for deadness (nested functions may be closures)
        if "." in info.qualname:
            continue
        # Exclude tests from dead-code reporting (pytest discovers by naming conventions)
        if kind_by_path.get(info.path) == "tests":
            continue
        if node_id in called_nodes:
            continue
        if f"{info.module_id}:{info.qualname}" in imported_by_name:
            continue
        if info.qualname in name_uses.get(info.module_id, set()):
            continue
        dead_funcs.append({
            "node_id": node_id,
            "path": info.path.relative_to(REPO_ROOT).as_posix(),
            "lineno": info.lineno,
        })

    dead_funcs = sorted(dead_funcs, key=lambda d: (d["path"], d["lineno"]))

    # Parse / syntax errors
    syntax_errors = [
        {"path": p.relative_to(REPO_ROOT).as_posix(), "error": err}
        for p, err in parse_errors.items()
        if err
    ]

    # Constant branches + redundant checks
    unreachable_branches: List[Dict[str, object]] = []
    redundant_checks: List[Dict[str, object]] = []
    nested_loops: List[Dict[str, object]] = []

    for p, cats in nested_loop_findings.items():
        rel_p = p.relative_to(REPO_ROOT).as_posix()
        if rel_p == "scripts/audit_repo.py":
            continue
        if rel_p.startswith("tests/"):
            continue
        for lineno, desc in cats.get("constant_branches", []):
            unreachable_branches.append({"path": p.relative_to(REPO_ROOT).as_posix(), "lineno": lineno, "desc": desc})
        for lineno, test_src in cats.get("redundant_checks", []):
            redundant_checks.append({"path": p.relative_to(REPO_ROOT).as_posix(), "lineno": lineno, "test": test_src})
        for lineno, desc in cats.get("nested_loops", []):
            nested_loops.append({"path": p.relative_to(REPO_ROOT).as_posix(), "lineno": lineno, "desc": desc})

    unreachable_branches.sort(key=lambda x: (x["path"], x["lineno"]))
    redundant_checks.sort(key=lambda x: (x["path"], x["lineno"]))
    nested_loops.sort(key=lambda x: (x["path"], x["lineno"]))

    # Over-engineering heuristics: "debug prints at import time" in library modules
    import_time_side_effects: List[str] = []
    for fi in files:
        if fi.kind != "src":
            continue
        text = fi.path.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"^print\(.*\)\s*$", text, flags=re.MULTILINE):
            # very rough; we know these files print at import
            if text.lstrip().startswith("print("):
                import_time_side_effects.append(fi.path.relative_to(REPO_ROOT).as_posix())

    report_lines: List[str] = []
    report_lines.append("# Repository Audit Report\n")
    report_lines.append("## Entry points\n")
    for k in ("operational", "utilities", "tests", "debug"):
        report_lines.append(f"- **{k}**")
        for p in entrypoints[k]:
            report_lines.append(f"  - {p.relative_to(REPO_ROOT).as_posix()}")
    report_lines.append("")

    report_lines.append("## Reachability\n")
    report_lines.append(f"- Operational reachable files: {len(operational_reach)} / {len(all_paths)}")
    report_lines.append(f"- Utilities reachable files: {len(utilities_reach)} / {len(all_paths)}")
    report_lines.append(f"- Tests reachable files: {len(tests_reach)} / {len(all_paths)}")
    report_lines.append("")

    report_lines.append("## Never executed (operational mode)\n")
    for rel in never_executed_operational_rel:
        report_lines.append(f"- {rel}")
    report_lines.append("")

    report_lines.append("## Never imported by any file (excluding entrypoints)\n")
    for rel in sorted(never_imported_anywhere):
        report_lines.append(f"- {rel}")
    report_lines.append("")

    report_lines.append("## Dead functions (best-effort)\n")
    for d in dead_funcs:
        report_lines.append(f"- {d['node_id']} ({d['path']}:{d['lineno']})")
    report_lines.append("")

    report_lines.append("## Duplicate functions (exact-structure matches)\n")
    if not duplicates:
        report_lines.append("- None detected by exact AST match")
    else:
        for dup in duplicates[:50]:
            report_lines.append(f"- {dup['count']} occurrences: {dup['signature']}")
            for occ in dup["occurrences"]:
                report_lines.append(f"  - {occ['node_id']} ({occ['path']}:{occ['lineno']})")
    report_lines.append("")

    report_lines.append("## Loop inefficiencies\n")
    for item in nested_loops:
        report_lines.append(f"- {item['path']}:{item['lineno']} {item['desc']}")
    for item in unnecessary_iterations:
        report_lines.append(f"- {item['path']}: {item['issue']} ({item['impact']})")
    report_lines.append("")

    report_lines.append("## Unreachable / redundant logic heuristics\n")
    if unreachable_branches:
        report_lines.append("### Constant branches\n")
        for item in unreachable_branches:
            report_lines.append(f"- {item['path']}:{item['lineno']} {item['desc']}")
    if redundant_checks:
        report_lines.append("\n### Repeated checks in same scope\n")
        for item in redundant_checks[:80]:
            report_lines.append(f"- {item['path']}:{item['lineno']} {item['test']}")
    report_lines.append("")

    report_lines.append("## Parse / syntax errors\n")
    if not syntax_errors:
        report_lines.append("- None")
    else:
        for e in syntax_errors:
            report_lines.append(f"- {e['path']}: {e['error']}")
    report_lines.append("")

    report_lines.append("## Import-time side effects\n")
    if not import_time_side_effects:
        report_lines.append("- None")
    else:
        for rel in sorted(import_time_side_effects):
            report_lines.append(f"- {rel}")
    report_lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "entrypoints": {k: [p.relative_to(REPO_ROOT).as_posix() for p in v] for k, v in entrypoints.items()},
        "reachability": {
            "operational": sorted([p.relative_to(REPO_ROOT).as_posix() for p in operational_reach]),
            "utilities": sorted([p.relative_to(REPO_ROOT).as_posix() for p in utilities_reach]),
            "tests": sorted([p.relative_to(REPO_ROOT).as_posix() for p in tests_reach]),
        },
        "never_executed_operational": never_executed_operational_rel,
        "never_imported_anywhere": sorted(never_imported_anywhere),
        "dead_functions": dead_funcs,
        "duplicates": duplicates,
        "loop_inefficiencies": {
            "nested_loops": nested_loops,
            "unnecessary_iterations": unnecessary_iterations,
        },
        "unreachable_branches": unreachable_branches,
        "redundant_checks": redundant_checks,
        "parse_errors": syntax_errors,
        "import_time_side_effects": import_time_side_effects,
    }


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/audit_report.md", help="Markdown report output path")
    args = parser.parse_args(argv)

    files = [classify_file(p) for p in iter_py_files()]

    entrypoints = find_entrypoints(files)
    local_edges, imported_modules_by_file, from_symbols_by_file, parse_errors = build_import_graph(files)

    funcs_by_id, call_edges, raw_calls, heuristics = build_call_graph(files)

    duplicates = detect_duplicate_functions(files)
    unnecessary_iterations = detect_unnecessary_iterations(files)

    out_md = (REPO_ROOT / args.out).resolve()

    audit_json = render_report(
        out_md=out_md,
        files=files,
        entrypoints=entrypoints,
        local_import_edges=local_edges,
        parse_errors=parse_errors,
        funcs_by_id=funcs_by_id,
        call_edges=call_edges,
        from_symbols_by_file=from_symbols_by_file,
        duplicates=duplicates,
        nested_loop_findings=heuristics,
        unnecessary_iterations=unnecessary_iterations,
    )

    out_dir = out_md.parent
    write_dot_import_graph(local_edges, out_dir / "import_graph.dot")
    write_dot_call_graph(call_edges, out_dir / "call_graph.dot")
    (out_dir / "audit.json").write_text(json.dumps(audit_json, indent=2), encoding="utf-8")

    print(f"Wrote report: {out_md.relative_to(REPO_ROOT).as_posix()}")
    print(f"Wrote graphs: {(out_dir / 'import_graph.dot').relative_to(REPO_ROOT).as_posix()}, {(out_dir / 'call_graph.dot').relative_to(REPO_ROOT).as_posix()}")
    print(f"Wrote data  : {(out_dir / 'audit.json').relative_to(REPO_ROOT).as_posix()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
