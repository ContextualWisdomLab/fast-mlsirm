# Descriptor-safe bounded JSON input

Repository automation treats JSON artifacts as untrusted local or CI inputs. All
governed readers delegate to `scripts._bounded_json.read_json_object`.

The loader:

1. opens the leaf with `O_CLOEXEC`, `O_NONBLOCK`, and `O_NOFOLLOW` when those
   platform flags are available;
2. validates the opened descriptor with `fstat` as a regular file;
3. compares the descriptor identity with `lstat` before and after the read;
4. reads at most 32 MiB plus one byte through that same descriptor;
5. scans structural nesting non-recursively with an inclusive depth limit of 128,
   ignoring delimiters inside strings and escaped characters;
6. decodes strict UTF-8, delegates syntax and value construction to `json.loads`,
   and requires an object root.

These controls bound availability risk and reject symbolic-link leaves, FIFOs,
directories, oversized inputs, excessive nesting, and detectable path
replacement. They do not make arbitrary JSON semantically trustworthy, replace
operating-system permission boundaries, or authorize processing from a hostile
writable parent directory.

## References

MITRE. (2026, April 30). *CWE-400: Uncontrolled resource consumption* (CWE
Version 4.20). https://cwe.mitre.org/data/definitions/400.html

MITRE. (2026, April 30). *CWE-674: Uncontrolled recursion* (CWE Version 4.20).
https://cwe.mitre.org/data/definitions/674.html

Python Software Foundation. (2026). *json—JSON encoder and decoder* (Python
3.12.13 documentation). https://docs.python.org/3.12/library/json.html

Python Software Foundation. (2026). *os—Miscellaneous operating system
interfaces* (Python 3.12.13 documentation).
https://docs.python.org/3.12/library/os.html
