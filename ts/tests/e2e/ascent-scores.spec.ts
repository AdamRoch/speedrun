// Copyright: Ankitects Pty Ltd and contributors
// License: GNU AGPL, version 3 or later; http://www.gnu.org/licenses/agpl.html

/**
 * Ascent Scores page smoke test.
 *
 * The screen shipped once without rendering at all (its Qt webview lacked the
 * kind that grants API access, so /_anki/ascentScores returned 403). Nothing
 * else exercises this page in a browser: the Rust tests stop below the
 * mediasrv boundary. This spec loads the real page against a live backend and
 * asserts a topic row renders for an AAMC-tagged card.
 *
 * Note: the e2e harness sets ANKI_API_HOST=0.0.0.0, which disables mediasrv's
 * bearer-token check, so the Qt-side kind→profile mapping itself is guarded
 * separately by qt/tests/test_webview_api_access.py.
 *
 * This test mutates the collection — a note is persisted on every run.
 */

import { Empty } from "@generated/anki/generic_pb";
import { AddNoteRequest, Note } from "@generated/anki/notes_pb";
import { NotetypeId, NotetypeNames } from "@generated/anki/notetypes_pb";

import { expect, test } from "./fixtures";
import { callRpc } from "./helpers";

const DEFAULT_DECK_ID = 1n;

test("ascent scores page renders a topic row for an AAMC-tagged card", async ({ page }) => {
    // Any served page provides a browser context to issue RPCs from.
    await page.goto("/ascent-scores", { waitUntil: "domcontentloaded" });

    const notetypeNames = NotetypeNames.fromBinary(
        await callRpc(page, "getNotetypeNames", new Empty()),
    );
    const basicId = notetypeNames.entries.find((entry) => entry.name === "Basic")?.id;
    if (basicId === undefined) {
        throw new Error("Expected stock Basic notetype in e2e profile");
    }

    const note = Note.fromBinary(
        await callRpc(page, "newNote", new NotetypeId({ ntid: basicId })),
    );
    note.fields[0] = "Ascent e2e front";
    note.fields[1] = "Ascent e2e back";
    // Maps to AAMC content category 1A via map_tags in rslib/src/ascent/mod.rs.
    note.tags = ["1A-Amino_Acids"];
    await callRpc(page, "addNote", new AddNoteRequest({ note, deckId: DEFAULT_DECK_ID }), 3);

    // The page loads its data once in its SvelteKit load function; reload to
    // pick up the note added above.
    await page.reload({ waitUntil: "domcontentloaded" });

    // A 403 (or any RPC failure) leaves the page without its header and table.
    await expect(page.locator("h1")).toHaveText("Ascent", { timeout: 15_000 });
    const row = page.locator("tbody tr").filter({ hasText: "1A" }).first();
    await expect(row).toBeVisible({ timeout: 15_000 });
    // Both estimate cells render something (a value or an explicit withhold).
    await expect(row.locator("td").nth(1)).not.toBeEmpty();
    await expect(row.locator("td").nth(2)).not.toBeEmpty();
});
