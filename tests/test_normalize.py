import pytest
from core.scoring.normalize import normalize_text, extract_canonical_tokens, matches_target

def test_normalize_unicode_nfkc_and_quotes():
    raw = "‘Project’  “Obsidian”\u2014Code: \uff36\uff29\uff2f\uff2c\uff25\uff34-7319"
    norm = normalize_text(raw)
    assert "project" in norm
    assert "obsidian" in norm
    assert "violet-7319" in norm
    assert "  " not in norm

def test_extract_canonical_tokens_preserves_hyphenated_codes():
    text = "The target key is OBSIDIAN-9482 and pin is PIN-7719."
    tokens = extract_canonical_tokens(text)
    assert "obsidian-9482" in tokens
    assert "pin-7719" in tokens

def test_matches_target_exact_and_substring():
    target = "acute hyperkalemia"
    resp_exact = "acute hyperkalemia"
    resp_sub = "The participant presented with acute hyperkalemia yesterday."
    resp_none = "The participant was completely healthy."

    m1, t1 = matches_target(resp_exact, target)
    assert m1 is True
    assert t1 == "exact"

    m2, t2 = matches_target(resp_sub, target)
    assert m2 is True
    assert t2 == "substring"

    m3, t3 = matches_target(resp_none, target)
    assert m3 is False
    assert t3 == "none"

def test_matches_target_alias():
    target = "OBSIDIAN-9482"
    aliases = ["Code Obsidian 9482", "OBS-9482"]
    resp = "Refer to authorization code OBS-9482 for clearance."
    matched, match_type = matches_target(resp, target, aliases=aliases)
    assert matched is True
    assert match_type == "alias"

def test_matches_target_token_overlap():
    target = "severe systemic acute inflammatory response syndrome"
    resp = "Patient exhibited systemic acute inflammatory response and fever."
    matched, match_type = matches_target(resp, target, threshold=0.5)
    assert matched is True
    assert match_type == "token_overlap"
