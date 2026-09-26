# Agent Instructions (AGENMTS.md)

This document outlines the coding standards, architectural principles, and toolchain rules for AI agents contributing to this repository.

## 1. Environment & Tooling
- **Python Version:** Strictly use **Python 3.12**. Do not use Python 3.13 or newer, as they introduce structural changes (e.g., experimental free-threaded GIL) that are not currently adopted for this project.
- **Package Management (uv):** This is a `uv`-based repository. 
  - When creating standalone scripts, use `uv run` and inline script metadata (PEP 723). 
  - When creating larger applications or libraries, initialize a `uv project`.
- **Linting & Formatting:** Use `ruff` as the standard linter and formatter. Ensure code complies with `ruff` rules to maintain a unified, modern codebase (taking advantage of the installed `ruff` extension).

## 2. Modern Python & Typing
- **Type Hints:** Adhere strictly to modern Python type hinting standards (PEP 585, PEP 604).
  - Prefer built-in generic types (`list`, `dict`, `set`) over legacy `typing` module imports (`List`, `Dict`, `Set`).
  - Use the pipe syntax for unions (`int | str`) instead of `Union`.
  - Only import from the `typing` module when absolutely necessary (e.g., `Any`, `Callable`).
  - Always prefer concrete types; use `Any` sparingly and only as a last resort.
- **Data Modeling:** This is a Pydantic-heavy project. Use `pydantic.BaseModel` to build strongly typed, validated data structures.

## 3. Architecture & Software Engineering
- **Paradigm:** Default to Object-Oriented Programming (OOP) where appropriate, structuring code into cohesive classes and modules.
- **SOLID Principles:**
  - **S**ingle Responsibility Principle: A class should have one, and only one, reason to change.
  - **O**pen/Closed Principle: Software entities should be open for extension, but closed for modification.
  - **L**iskov Substitution Principle: Objects in a program should be replaceable with instances of their subtypes without altering the correctness of that program.
  - **I**nterface Segregation Principle: Many client-specific interfaces are better than one general-purpose interface.
  - **D**ependency Inversion Principle: Depend upon abstractions, not concretions.
- **DRY (Don't Repeat Yourself):** Abstract duplicated logic into reusable functions, methods, or base classes.
- **Design Patterns:** Proactively employ industry-standard design patterns to solve common architectural problems (e.g., Factory, Builder, Facade, Adapter, Strategy, Singleton, Observer, etc.).
- **Single Source of Truth:** When creating configs and submodules, follow the rule of single source of truth - e.g avoid default typing for every config class implemented as pydantic and have a single "wrapping" config object that has all the submodules and ONLY IN IT set all the specific types. This approach would help to control every parameter in a single object, and avoid the need to import multiple config classes from different places.
- **Anti-Magic:** Strictly avoid "magic numbers" and "magic strings" scattered throughout the logic. Extract configuration values, model paths, dataset names, and thresholds into module-level constants or Pydantic configuration objects. 
  - *Exception 1 (Idiomatic Python):* Standard structural numbers are perfectly fine (e.g., `list[-1]`, `if count > 0`, `return 1`).
  - *Exception 2 (Visualization & CLI):* Highly specific, ad-hoc string labels and layout numbers in data visualization libraries (like `matplotlib`, `seaborn`, `plotly`) or CLI flag descriptions (like `argparse`) do not need to be extracted into constants.
- **Pythonic Namespacing (No Stateless Classes):** Avoid creating classes that only contain `@staticmethod` or `@classmethod` if the class has no state. In Python, modules themselves are the namespaces. Use module-level functions for stateless operations (like factories) instead of empty classes.
- **Short Naming Convention:** Do not repeat the parent directory name in the file name. For example, use `models/audio.py` instead of `models/audio_model.py`, and `factories/pipeline.py` instead of `factories/pipeline_factory.py`.

## 4. Execution & Flow Control
- **Error Handling & Exceptions:** Avoid bare `except:` blocks. Always catch specific exceptions. Define custom exception classes that inherit from a base project exception where appropriate.
- **Asynchronous Execution:** When dealing with high-volume I/O or network requests, favor `asyncio` (and libraries like `aiohttp` or async `httpx`) over synchronous execution.

## 5. Code Quality, Testing, & Logging
- **Efficiency vs. Readability:** Strive for highly efficient and clean code, but **never at the expense of readability**. Clever, overly terse code is an anti-pattern if it is difficult for humans to parse.
- **Testing Standards:** 
  - **Strictly `pytest`:** Use only `pytest`. The built-in `unittest` package is strictly forbidden.
  - **Assertions with `assertpy`:** Utilize the `assertpy` library for all test assertions (e.g., `assert_that`, `soft_assertions`) rather than standard Python `assert` statements.
  - **Data-Driven Testing:** Make tests as data-driven as possible. Write generalized test functions and feed them varying inputs and expected outputs using `@pytest.mark.parametrize` to eliminate repetitive, hardcoded test cases.
  - **Meaningful Coverage Only:** Do not blindly add tests just to inflate the test count. Only write tests that guard critical logic, edge cases, or complex data transformations. Whenever you refactor or add new logic, proactively assess if a new *meaningful* test is applicable to protect the system's structural integrity.
- **Structured Logging:** Do not use scattered `print()` statements for debugging or operational flow. Default to a standard logger (like the built-in `logging` module or `loguru`) and include contextual information for easier debugging.
- **Documentation:** Write clear, concise comments and docstrings. Provide the minimal description needed for future reference. Avoid over-commenting obvious logic; let the code speak for itself through expressive variable and function names.
- **Imports:** Use direct imports and avoid relative imports for all internal imports. Avoid using `__all__` in `__init__.py` files unless explicitly exporting a well-defined public API.
## 6. Agent Workflow & Hygiene
- **Temporary Scripts & Artifacts:** Never write scratch scripts, test data, or temporary exploratory files directly to the repository root or project folders. ALWAYS use a temporary directory (like `/tmp/` or the agent's `scratch/` artifact directory).
- **Repository Cleanliness:** If you absolutely must create temporary files in the repository to debug something, you MUST delete them and clean up after yourself before completing your turn. Do not leave behind uncommitted temporary artifacts.
- **Git Restrictions:** NEVER execute `git commit` or `git push` under any circumstances unless explicitly, unambiguously requested by the user. Modify local files only, and leave the staging/committing/pushing process strictly to the human reviewer.
- **No Config Mutation (No Injection):** Never mutate configuration objects (like Pydantic models) after they are initialized. All values must be passed during the object's initialization constructor. Mutating state post-initialization (`config.dataset.path = "..."`) is a strict code smell.
- **Refactoring Scope & Tracing:** Before concluding any refactor (changing class signatures, method names, config fields, etc.), you MUST perform a codebase-wide search (e.g., using `grep`) to identify and update all callers, instantiations, and dependencies. Never assume a change is isolated.
