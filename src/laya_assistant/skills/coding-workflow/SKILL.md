---
name: coding-workflow
description: "How to write, run and verify code in the sandbox: write the file, run it or its tests, read failures, fix, re-run. Use for any request to write, fix or refactor code."
---

# coding-workflow

1. Write the code with `write_file` using a relative path (for example `palindrome.py`).
2. Write tests next to it (`test_palindrome.py`) when the request implies correctness matters.
3. Verify with `execute`: `python -m pytest -q` (no path argument) or `python palindrome.py`.
4. If a command fails, read the LAST lines of the output, change the code, and re-run. Never repeat the identical failing command.
5. Finish with two sentences: what you built and how to run it. Do not paste the whole file back.
