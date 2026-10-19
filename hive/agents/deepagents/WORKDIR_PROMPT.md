## Working directory

Your working directory is `{workdir}`. The file tools (`ls`, `read_file`, `write_file`, `edit_file`, `delete`, `glob`, `grep`) treat `/` as that directory, so `/solution.py` means `{workdir}/solution.py`; the full path `{workdir}/solution.py` works too. Shell commands run by `execute` start in `{workdir}`.
