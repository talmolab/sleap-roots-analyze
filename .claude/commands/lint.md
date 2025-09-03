Run linting with `ruff`.

Command:

```
uv run ruff format src/sleap_roots_analyze tests && uv run ruff check --fix src/sleap_roots_analyze tests
```

Then manually fix any remaining errors which cannot be automatically fixed by ruff.