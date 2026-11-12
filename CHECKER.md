# Installation checker (review issue 27)

Serve `check_install_JS.html` through a local web server, then choose the main
HTML/JS file to scan. HTML attributes and inline scripts/styles are inspected
separately; JavaScript recognizes static imports/re-exports, side-effect imports,
workers/shared workers, importScripts and fetch, including literal new URL paths
relative to import.meta.url. CSS supports url(...) and @import references.

The scanner chooses recursive text parsing from the URL pathname, preserving
query parameters in requests while ignoring fragments for duplicate detection.
It reports dynamic paths, bare module specifiers, import maps and srcset for
manual verification. It is a static dependency aid, not a complete JavaScript
parser or execution-based dependency tracer.

Run `node --test scripts/checker.test.cjs` for scanner and recursive-request checks.
The workspace-root checker is updated to the same implementation.
