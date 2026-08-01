# Copyright: Ankitects Pty Ltd and contributors
# License: GNU AGPL, version 3 or later; http://www.gnu.org/licenses/agpl.html

"""Entry point for `just probe-gen`; see anki.probe_gen."""

import sys

sys.path.extend(["pylib", "out/pylib"])

# anki.collection must load before anki.probe_gen's `from anki.cards import`
# runs, or the cards<->hooks_gen import cycle bites.
import anki.collection  # noqa: E402, F401
from anki.probe_gen import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
