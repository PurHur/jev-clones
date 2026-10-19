"""Negative controls for the card suite: a mutated copy of README.md must make the static tier FAIL, so a suite that
passes is known to be able to fail. Each control runs `pytest test_card.py -m static` in a subprocess with
CARD_README_OVERRIDE pointing at the mutated card (the commit's files are untouched) and expects a non-zero exit with
the named test among the failures.

    CARD_REV=<40-hex> CARD_DIR=<folder> pytest test_card_controls.py
"""
import os
import re
import subprocess
import sys
import tempfile
import unittest

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CONTROLS = [
    # (name, mutation of the card text, the test that must fail)
    ("a registered number changed", lambda t: t.replace("**203 / 231 (87.9 %)**", "**213 / 231 (92.2 %)**", 1), "test_registered_claims_are_in_the_card"),
    ("a file the card names does not exist", lambda t: t.replace("`serve_config.json`", "`serve_config_v2.json`", 1), "test_every_file_the_card_names_exists"),
    ("a tag the card names does not exist", lambda t: t.replace("`--revision v1.2.3`", "`--revision v1.2.9`", 1), "test_pins_and_tags_resolve"),
    ("the hybrid agreement sentence removed", lambda t: re.sub(r"hybrid-mode agreement with the merged v1\.3(?:\.1)? weights on the same 231 items is \d+ of 231", "hybrid-mode agreement with the merged v1.3 weights is high", t), "test_registered_claims_are_in_the_card"),
    ("a json example made unparsable", lambda t: t.replace('"probabilities": {"p1": 0.948, "p2": 0.029, "p3": 0.023}},', '"probabilities": {"p1": 0.948, "p2": 0.029, "p3": 0.023},', 1), "test_json_blocks_parse_and_examples_pair_up"),
]


@pytest.mark.static
class NegativeControls(unittest.TestCase):
    def run_static(self, text):
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(text); path = f.name
        env = dict(os.environ, CARD_README_OVERRIDE=path, CARD_REPORT=path + ".json", CARD_OFFLINE=os.environ.get("CARD_OFFLINE", ""))
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_card.py", "-m", "static"], cwd=HERE, env=env,
                           capture_output=True, text=True, timeout=900)
        os.remove(path)
        if os.path.exists(path + ".json"):
            os.remove(path + ".json")
        return r

    def test_the_unmutated_card_passes(self):
        folder = os.environ.get("CARD_DIR") or HERE
        text = open(os.path.join(folder, "README.md"), encoding="utf-8").read()
        r = self.run_static(text)
        self.assertEqual(r.returncode, 0, r.stdout[-1500:])

    def test_each_mutation_fails_the_named_test(self):
        folder = os.environ.get("CARD_DIR") or HERE
        text = open(os.path.join(folder, "README.md"), encoding="utf-8").read()
        for name, mutate, failing in CONTROLS:
            with self.subTest(control=name):
                mutated = mutate(text)
                self.assertNotEqual(mutated, text, f"the mutation '{name}' did not change the card: update the control")
                r = self.run_static(mutated)
                self.assertNotEqual(r.returncode, 0, f"the static tier passed on a mutated card ({name})")
                self.assertIn(failing, r.stdout, f"{failing} did not fail on '{name}':\n{r.stdout[-1200:]}")
