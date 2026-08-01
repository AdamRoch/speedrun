# Copyright: Ankitects Pty Ltd and contributors
# License: GNU AGPL, version 3 or later; http://www.gnu.org/licenses/agpl.html

"""Guards the webview kind → API access mapping.

The Ascent Scores screen once shipped with a webview kind missing from the
allow-list: no Authorization header was attached, every RPC returned 403, and
the dialog rendered a Flask error page. The e2e harness disables the server
side of this check (ANKI_API_HOST=0.0.0.0), so the mapping is pinned here.
"""

from __future__ import annotations

import inspect


def test_ascent_scores_kind_has_api_access() -> None:
    from aqt.webview import KINDS_WITH_API_ACCESS, AnkiWebViewKind

    assert AnkiWebViewKind.ASCENT_SCORES in KINDS_WITH_API_ACCESS


def test_ascent_dialog_uses_api_access_kind() -> None:
    # The dialog needs a running app to instantiate, so pin the constructor
    # source instead: it must select a kind that carries API access, not the
    # add-on-style title= path that silently downgrades to DEFAULT.
    from aqt.ascent import AscentScoresDialog

    source = inspect.getsource(AscentScoresDialog.__init__)
    assert "kind=AnkiWebViewKind.ASCENT_SCORES" in source
