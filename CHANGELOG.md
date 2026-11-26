# Changelog

What changed in each release, and why. Newest first.

## 0.2.0 (unreleased): vsl-core 0.4, no F2 claims

**Before:** vsl-langgraph required vsl-core `>=0.3,<0.4`. Every example, doc snippet and test
passed `assurance_basis=AssuranceBasis(f1_pre_commitment=True, f2_modification=F2Modification.FULL)`,
which overclaimed for gates that only allow or block, and `LangGraphAdapter` scanned a monitor's or
rule's source to warn when it claimed `FULL` while calling a hosted model API.

**After:** vsl-langgraph requires vsl-core `>=0.4,<0.5`. Gates are built from a name, description
and monitor or rule, with no assurance claim. The FULL-claim heuristic, its test and its two test
fixtures are gone: vsl-core 0.4 takes no F2 value, so there is nothing to check. `docs/concepts.md`
describes the computed `Enforcement` in place of `AssuranceBasis`, and shows recording the guarded
node's side effect with `write_action()`. A new test runs a real `StateGraph` and checks the ledger
reads `BEFORE_ACTION` for an allowed and a denied call.

**Why:** vsl-core 0.4.0 computes whether a gate held before its action from the ledger instead of
taking a declared level. Code still passing `assurance_basis` keeps working, with a
`FutureWarning` from vsl-core.

## 0.1.2 (2026-11-13, tag `v0.1.2`): published on PyPI

The first version on PyPI: `pip install vsl-langgraph`, which installs vsl-core (`super-semantics-vsl`)
with it. The code is unchanged from 0.1.1.

**Before:** vsl-core came from a GitHub URL pinned to `v0.3.0`, which PyPI does not accept in a
published package, and an application that also installed vsl-core had to name the same URL.

**After:** vsl-core is a normal dependency, `super-semantics-vsl>=0.3,<0.4`, resolved from PyPI; the
README gives `pip install vsl-langgraph`, its links point to GitHub, and the package lists its repository.
A pushed release tag publishes from GitHub Actions through PyPI Trusted Publishing
(`.github/workflows/publish.yml`), after the tests pass.

**Why:** installs from a registry, with versions an application can share with its own dependencies.

## 0.1.1 (2026-11-12, tag `v0.1.1`): vsl-core pinned to v0.3.0

**Before:** vsl-core was installed from `git+https://github.com/supersemantics/VSL-Core.git`
with no version, so an install took whatever vsl-core's `main` was that day. An application could
not pin vsl-core itself either: pip refuses two different URLs for one package, so pinning
`@v0.3.0` next to this adapter failed to install.

**After:** vsl-core is pinned to its release tag, `...VSL-Core.git@v0.3.0`. An application that
also installs vsl-core names the same URL, tag included.

**Why:** reproducible installs; tested against vsl-core 0.3.0.

**Also:** the docs no longer point to a separate hosted ledger package; they name the ledger stores this library's core ships with.

## 0.1.0

The first release.
