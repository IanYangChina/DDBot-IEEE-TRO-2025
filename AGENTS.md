# AGENTS.md

> Purpose: define the agent hierarchy, responsibilities, workflows and deliverables for a **Copilot CLI research-coding assistant** whose job is to *clean and reconstruct* a code + LaTeX repository from a recent publication and make it ready for open-source release.

---

# 1. Summary / Goals

* Produce a clean, well-structured, documented, and reproducible public repository containing:

  * reproducible code for experiments,
  * the (compiled) paper and a mapping from paper elements (figures, tables) → code/data,
  * clear README and contribution instructions,
  * license and minimal packaging for reuse.
* Use three cooperating agents:

  1. **Co-Manager (coordinator & reviewer)** — orchestrates work, integrates results, writes final README intro/instructions, presents restructuring plan to user for approval.
  2. **LaTeX / Paper Reading Agent** — reads the paper(s), figures and supplementary material to extract what was published and map experiments → code/data/files.
  3. **Project Content Review & Edit Agent** — performs code cleaning, refactor, tests, packaging, documentation, and creates PRs / commits.

---

# 2. Agent roles & responsibilities

## 2.1 Co-Manager (Research Co-Manager) (Main agent)

**Primary responsibilities**

* Accepts repository root as input and launches subagents.
* Collects and synthesizes outputs from the Paper Reading and Project Review agents.
* Creates a clear, actionable repository restructuring plan (directory layout, files to keep/remove/rename, scripts to add).
* Presents the plan in a human-readable review to the user and requests explicit approval (yes / changes requested).
* After approval, orchestrates the Project Content Review & Edit agent to implement the plan.
* Writes the public-facing `README.md` introduction and usage examples (final copy), including:

  * Project description,
  * quickstart (install + run),
  * reproduction steps for main results,
  * citation instructions,
  * license and contact info.
* Verifies completion criteria and merges final PR(s) once tests/CI pass.

**Expected outputs**

* `plan/RESTRUCTURE_PLAN.md` (documented plan + file-level actions).
* `README.md` (final public intro & instructions).
* Merge-ready PR(s) with a clear commit history and PR description.

## 2.2 LaTeX / Paper Reading Agent (Subagent)

**Primary responsibilities**

* Parse LaTeX sources, PDF, and supplementary files to extract:

  * list of experiments, hyperparameters, datasets, and figures/tables,
  * which code files / scripts correspond to each experiment / figure,
  * any missing assets (data, models) referenced but not present,
  * build / compilation commands used in the paper (if present).
* Produce a mapping file: `paper_mapping/paper_to_code_map.yaml` or `.md` listing for each figure/table:

  * figure id/name → script(s) → input data → generated artifact(s)
* Flag ambiguous or missing items for the Co-Manager (e.g., “figure 3 uses `train.py` but hyperparams unspecified”).

**Expected outputs**

* `paper_mapping/paper_to_code_map.yaml` (or markdown).
* `paper_notes/MISSING_ASSETS.md` with a prioritized list of missing files and recommended actions.
* Short summary `paper_summary/summary.md` with critical reproduction steps.

## 2.3 Project Content Review & Edit Agent (Subagent)

**Primary responsibilities**

* Propose restructure plan for the Co-Manager to review and edit
* Implement the revised restructure plan produced by Co-Manager after user approval.
* Clean and edit code for clarity, reproducibility:

  * normalize repository layout,
  * add `pyproject.toml` (or `setup.py`) where appropriate,
  * add a `requirements.txt` or `environment.yml` and pin minimal versions,
  * add small reproducible examples (scripts or notebooks) for main experiments,
  * add unit / integration tests for core components,
  * format code (e.g., `black` / `isort`) and add `pre-commit` config,
  * add or improve docstrings, inline usage, minimal API docs,
  * convert broken or ad-hoc scripts into named CLI commands where appropriate
* Remove unnecessary artifacts (IDE files, large intermediate binaries) or move them to `/archive` with `README_archive.md`.

**Expected outputs**

* PR(s) implementing restructure and code cleanup.
* `tests/` with at least smoke tests for reproducibility.
* `setup` and `requirements`.

---

# 3. Tools & permissions (recommended)

Available capabilities:

* **Shell** — run commands (build, test, formatting).
* **Read files** — read repository files, LaTeX, PDFs, data manifests.
* **Edit files** — create/modify files, commit changes.
* **Subagents** — spawn / communicate with other agents (Co-Manager spawns subagents).
* **Ask user** — present plans and request approval or clarifications.
* **Web** (restricted) — only when necessary to look up dependency versions, dataset links, or citation metadata. Use minimal web access and log any lookups.

Agent abilities by role:

* Co-Manager: Shell, Read files, Edit files, Subagents, Ask user, Web (optional).
* Paper Reading Agent: Read files, Shell, Ask user (for ambiguous items), Web (to fetch external dataset links).
* Project Content Review Agent: Shell, Read files, Edit files, Ask user (for policy / scope decisions), Web (to fetch package versions, CI guides).

---