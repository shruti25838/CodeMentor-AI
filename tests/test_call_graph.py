"""The call and import graph, on small fixtures where the right answer is obvious.

Every fact here is read from the syntax tree; no model is involved anywhere in this module.
The tests state both what the graph resolves and what it deliberately leaves unresolved.
"""

import pytest

from codeatlas.services.analysis.call_graph import WHAT_IT_MISSES, CallGraph


def write(root, files: dict[str, str]):
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return CallGraph.build(root)


@pytest.fixture
def pkg(tmp_path):
    """A small package: helpers, a class hierarchy, and a module that uses both."""
    return write(
        tmp_path,
        {
            "mypkg/__init__.py": "from .core import Engine\nfrom .util import helper\n",
            "mypkg/util.py": ("def helper(x):\n    return trim(x)\n\ndef trim(x):\n    return x.strip()\n"),
            "mypkg/core.py": (
                "from .util import helper\n"
                "import mypkg.util as u\n"
                "\n"
                "class Base:\n"
                "    def describe(self):\n"
                "        return 'base'\n"
                "\n"
                "class Engine(Base):\n"
                "    def run(self, value):\n"
                "        cleaned = helper(value)\n"
                "        return self.describe() + cleaned\n"
                "\n"
                "    def other(self):\n"
                "        return u.trim('x')\n"
                "\n"
                "def make():\n"
                "    return Engine()\n"
            ),
        },
    )


# ---------- where is X defined ----------


def test_where_is_a_class_defined(pkg) -> None:
    [found] = pkg.where_is_it_defined("Engine")
    assert found.path == "mypkg/core.py"
    assert found.kind == "class"
    assert found.qualname == "mypkg.core.Engine"


def test_where_is_a_method_defined(pkg) -> None:
    [found] = pkg.where_is_it_defined("Engine.run")
    assert found.kind == "method"
    assert found.parent == "mypkg.core.Engine"
    assert found.start_line == 9


def test_a_name_defined_once_has_one_location(pkg) -> None:
    assert len(pkg.where_is_it_defined("trim")) == 1


def test_a_name_defined_twice_has_both_locations(tmp_path) -> None:
    graph = write(tmp_path, {"a.py": "def go():\n    pass\n", "b.py": "def go():\n    pass\n"})
    assert {d.path for d in graph.where_is_it_defined("go")} == {"a.py", "b.py"}


def test_an_unknown_name_has_no_location(pkg) -> None:
    assert pkg.where_is_it_defined("nope") == []


# ---------- who calls X ----------


def test_who_calls_a_module_level_function(pkg) -> None:
    callers = {c.caller for c in pkg.who_calls("helper")}
    assert callers == {"mypkg.core.Engine.run"}


def test_who_calls_reports_the_line(pkg) -> None:
    [call] = pkg.who_calls("helper")
    assert call.line == 10
    assert call.path == "mypkg/core.py"


def test_a_call_through_self_resolves_to_the_inherited_method(pkg) -> None:
    """self.describe() in Engine resolves to Base.describe."""
    callers = {c.caller for c in pkg.who_calls("Base.describe")}
    assert callers == {"mypkg.core.Engine.run"}


def test_a_call_through_an_aliased_module_resolves(pkg) -> None:
    """u.trim() where `import mypkg.util as u`."""
    callers = {c.caller for c in pkg.who_calls("trim")}
    assert "mypkg.core.Engine.other" in callers


def test_calling_a_class_resolves_to_the_class(pkg) -> None:
    callers = {c.caller for c in pkg.who_calls("Engine")}
    assert callers == {"mypkg.core.make"}


def test_nobody_calls_an_uncalled_function(pkg) -> None:
    assert pkg.who_calls("make") == []


# ---------- what does X call ----------


def test_what_a_method_calls(pkg) -> None:
    calls = pkg.what_does_it_call("Engine.run")
    assert {c.callee_text for c in calls} == {"helper", "self.describe"}
    assert all(c.resolved for c in calls)


def test_unresolved_calls_are_kept_and_marked(pkg) -> None:
    """trim() calls x.strip(), which is a method on a value of unknown type."""
    calls = pkg.what_does_it_call("trim")
    [strip] = [c for c in calls if c.callee_text == "x.strip"]
    assert not strip.resolved
    assert strip.callee == ""


def test_a_function_that_calls_nothing(tmp_path) -> None:
    graph = write(tmp_path, {"a.py": "def quiet():\n    return 1\n"})
    assert graph.what_does_it_call("quiet") == []


# ---------- imports ----------


def test_who_imports_a_module(pkg) -> None:
    assert {e.module for e in pkg.who_imports("mypkg.util")} == {"mypkg", "mypkg.core"}


def test_what_a_module_imports(pkg) -> None:
    internal = {e.resolved_module for e in pkg.what_does_it_import("mypkg.core") if e.internal}
    assert internal == {"mypkg.util"}


def test_a_relative_import_resolves_against_the_importing_package(pkg) -> None:
    [edge] = [e for e in pkg.what_does_it_import("mypkg.core") if e.name == "helper"]
    assert edge.target == ".util"
    assert edge.resolved_module == "mypkg.util"


def test_a_package_init_resolves_its_own_relative_imports(pkg) -> None:
    """`from .core import Engine` in mypkg/__init__.py is relative to mypkg itself."""
    internal = {e.resolved_module for e in pkg.what_does_it_import("mypkg") if e.internal}
    assert internal == {"mypkg.core", "mypkg.util"}


def test_a_standard_library_import_is_not_matched_to_a_repo_module(tmp_path) -> None:
    """`import logging` must not resolve to this repo's own logging.py."""
    graph = write(tmp_path, {"pkg/__init__.py": "", "pkg/logging.py": "", "pkg/app.py": "import logging\n"})
    [edge] = graph.what_does_it_import("pkg.app")
    assert edge.target == "logging"
    assert not edge.internal


def test_an_external_import_is_recorded_but_not_internal(pkg) -> None:
    graph = pkg
    edges = [e for e in graph.imports if e.target == "os"]
    assert all(not e.internal for e in edges)


# ---------- circular imports ----------


def test_no_cycle_is_reported_when_there_is_none(pkg) -> None:
    assert pkg.circular_imports() == []


def test_a_two_module_cycle_is_found(tmp_path) -> None:
    graph = write(
        tmp_path,
        {"p/__init__.py": "", "p/a.py": "from .b import thing\n", "p/b.py": "from .a import other\n"},
    )
    assert graph.circular_imports() == [["p.a", "p.b"]]


def test_a_longer_cycle_is_found_as_one_group(tmp_path) -> None:
    graph = write(
        tmp_path,
        {
            "p/__init__.py": "",
            "p/a.py": "from .b import x\n",
            "p/b.py": "from .c import y\n",
            "p/c.py": "from .a import z\n",
            "p/alone.py": "import os\n",
        },
    )
    assert graph.circular_imports() == [["p.a", "p.b", "p.c"]]


def test_the_shortest_cycle_through_a_module_is_reported(tmp_path) -> None:
    graph = write(
        tmp_path,
        {"p/__init__.py": "", "p/a.py": "from .b import x\n", "p/b.py": "from .a import y\n"},
    )
    assert graph.shortest_import_cycle("p.a") == ["p.a", "p.b", "p.a"]


def test_no_cycle_through_a_module_that_is_not_in_one(pkg) -> None:
    assert pkg.shortest_import_cycle("mypkg.util") == []


# ---------- the limits are stated, and real ----------


def test_a_dynamic_call_is_left_unresolved(tmp_path) -> None:
    graph = write(
        tmp_path,
        {"a.py": "def target():\n    pass\n\ndef go(name):\n    fn = globals()[name]\n    return fn()\n"},
    )
    assert graph.who_calls("target") == []
    assert any(not c.resolved for c in graph.what_does_it_call("go"))


def test_a_call_on_a_variable_of_unknown_type_is_unresolved(tmp_path) -> None:
    graph = write(
        tmp_path,
        {"a.py": "class K:\n    def m(self):\n        pass\n\ndef go(obj):\n    return obj.m()\n"},
    )
    assert graph.who_calls("K.m") == []


def test_the_module_documents_what_it_misses() -> None:
    for topic in ("Dynamic calls", "Decorators", "Inheritance", "import *", "Only Python"):
        assert topic in WHAT_IT_MISSES


def test_building_an_empty_tree_is_not_an_error(tmp_path) -> None:
    graph = CallGraph.build(tmp_path)
    assert graph.definitions == {}
    assert graph.calls == []
    assert graph.circular_imports() == []


def test_a_file_that_does_not_parse_cleanly_does_not_stop_the_build(tmp_path) -> None:
    graph = write(tmp_path, {"good.py": "def ok():\n    pass\n", "bad.py": "def (((\n"})
    assert graph.where_is_it_defined("ok")


def test_counts_are_reported(pkg) -> None:
    resolved, total = pkg.resolution_rate()
    assert 0 < resolved <= total
