import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import common
import compare_activations
import decompose_loss
import detect_entities
import edit_components
import restore_activations
import retrieve_sentences
import rotate_encoder


def geometry():
    rng = np.random.default_rng(42)
    native = rng.normal(size=(24, 4))
    mean = np.array([0.1, -0.2, 0.3, 0.4])
    eigenvalues = np.array([4., 3., 2., 1.])
    beta = np.array([[2., 1.], [1., 2.], [1., -1.], [0.5, 0.4]])
    weight = beta / np.sqrt(eigenvalues)[:, None]
    return dict(native=native, mean=mean, eigenvalues=eigenvalues,
                eigenvectors=np.eye(4), basis=np.eye(4), beta=beta,
                pre=native @ weight + np.array([0.5, -0.3]), thresholds=np.array([0.2, 0.4]),
                labels=np.tile([0, 1], 12), feature_ids=np.array([17, 29]),
                angle=np.array([0.05, 0.05]), reference=np.array([0.05, 0.05]))


class AnalysisTests(unittest.TestCase):
    def test_auroc_ties_and_missing_classes(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            positive, negative = rng.integers(-3, 4, (2, 10))
            expected = ((positive[:, None] > negative).sum()
                        + 0.5 * (positive[:, None] == negative).sum()) / 100
            self.assertEqual(common.auroc(positive, negative), expected)
        self.assertTrue(np.isnan(common.auroc(np.array([]), np.array([1.]))))

    def test_threshold_boundary_and_loss_closure(self):
        pre = np.array([[-2., 2.], [0., -1.], [0.5, 0.], [1., 1.], [2., 3.], [3., 2.]])
        threshold = np.array([1., 2.])
        post = common.jumprelu(pre, threshold)
        self.assertEqual(post[3, 0], 0.)
        self.assertEqual(post[0, 1], 0.)
        for row in compare_activations.compare(pre, threshold, np.tile([0, 1], 3)):
            self.assertAlmostEqual(row["total_loss"], row["nonpositive_loss"] + row["positive_loss"])

    def test_restoration_matches_decoder_edit(self):
        data = geometry()
        rng = np.random.default_rng(3)
        residual, decoder = rng.normal(size=(24, 3)), rng.normal(size=(2, 3))
        added, _ = restore_activations.restoration(data["pre"], data["thresholds"])
        actual = restore_activations.gain_formula(residual, added, decoder)
        expected = (np.square(residual).sum(1)[:, None]
                    - np.square(residual[:, None, :] - added[:, :, None] * decoder[None]).sum(2))
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_rotation_norm_angle_and_forward(self):
        data = geometry()
        changed, _, _ = rotate_encoder.matched_edit(data["beta"], data["basis"],
                                                    data["reference"], data["angle"], 1)
        rotate_encoder.geometry_audit(data["beta"], changed, data["angle"])
        random = rotate_encoder.control(data["beta"], data["feature_ids"], data["angle"], 14801)
        rotate_encoder.geometry_audit(data["beta"], random, data["angle"])
        delta = (changed - data["beta"]) / np.sqrt(data["eigenvalues"])[:, None]
        expected = data["pre"] + (data["native"] - data["mean"]) @ delta
        np.testing.assert_allclose(common.edited_scores(data, changed), expected, atol=1e-12)

    def test_component_edits(self):
        data = geometry()
        top, rest, conditions, _ = edit_components.components(
            data["beta"], data["eigenvalues"], data["angle"], 1)
        np.testing.assert_allclose(top + rest, data["beta"])
        for name in ("combined", "rest_scaled"):
            np.testing.assert_allclose(np.linalg.norm(conditions[name], axis=0),
                                       np.linalg.norm(data["beta"], axis=0))
        self.assertTrue(np.all(conditions["neither"] == 0))

    def test_entity_cutoff_is_validation_optimum(self):
        labels = np.array([1, 0, 1, 0], dtype=bool)
        for scores in (np.zeros(4, dtype=np.float32), np.array([-2., -1., 0., 4.], dtype=np.float32)):
            cutoff = detect_entities.cutoff(scores, labels)
            candidates = np.r_[np.unique(scores).astype(np.float64),
                                np.nextafter(float(scores.min()), -np.inf)]
            def f1(value):
                prediction = scores.astype(np.float64) > value
                return 2 * (prediction & labels).sum() / (prediction.sum() + labels.sum())
            self.assertEqual(f1(cutoff), max(map(f1, candidates)))

    def test_retrieval_matches_materialized_resamples(self):
        scores = np.array([1., 1., 0., 3., 0., 2.])
        truth = np.array([1, 0, 1, 1, 0, 1], dtype=bool)
        ids = np.array(["b", "a", "f", "e", "c", "d"])
        order = retrieve_sentences.ranking(scores, np.argsort(ids))
        np.testing.assert_array_equal(order, [3, 5, 1, 0, 4, 2])
        weights = np.array([[1, 1, 1, 1, 1, 1], [2, 1, 0, 0, 1, 2]])
        actual = retrieve_sentences.bootstrap_tp(order, truth, weights, 3)
        expected = [truth[np.repeat(order, draw[order])[:3]].sum() for draw in weights]
        np.testing.assert_array_equal(actual, expected)

    def test_all_command_line_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            data = geometry()
            data.update(residual=np.ones((24, 3)), decoder=np.ones((2, 3)))
            path = folder / "geometry.npz"
            np.savez(path, **data)
            commands = [("compare_activations.py", path, []), ("restore_activations.py", path, []),
                        ("rotate_encoder.py", path, ["--k", "1", "--controls", "1"]),
                        ("edit_components.py", path, ["--k", "1", "--controls", "1"])]
            entity = folder / "entity.npz"
            np.savez(entity, validation_pre=data["pre"], test_pre=data["pre"],
                     thresholds=data["thresholds"], validation_labels=np.eye(2)[data["labels"]],
                     test_labels=np.eye(2)[data["labels"]], feature_ids=data["feature_ids"],
                     validation_feature_ids=data["feature_ids"], test_feature_ids=data["feature_ids"])
            commands.append(("detect_entities.py", entity, []))
            retrieval = folder / "retrieval.npz"
            post = common.jumprelu(data["pre"], data["thresholds"])
            np.savez(retrieval, scores=np.stack([post[None], post[None]]),
                     labels=np.eye(2)[data["labels"]], sentence_ids=np.array([str(i) for i in range(24)]))
            commands.append(("retrieve_sentences.py", retrieval, ["--budget", "3", "--draws", "10"]))
            loss = folder / "loss.npz"
            arrays = {}
            for coder in ("sae", "tc"):
                arrays.update({f"{coder}_rates": np.array([[0.8, 0.9], [0.6, 0.7]]),
                               f"{coder}_pre": np.array([0.8, 0.75]),
                               f"{coder}_relu": np.array([0.6, 0.6]),
                               f"{coder}_post": np.array([0.55, 0.55])})
            np.savez(loss, **arrays)
            commands.append(("decompose_loss.py", loss, []))
            for script, source, extra in commands:
                output = folder / (script + ".csv")
                subprocess.run([sys.executable, str(ROOT / script), str(source), "--output", str(output), *extra],
                               check=True, capture_output=True, text=True)
                self.assertTrue(output.exists())


@unittest.skipUnless(os.environ.get("ORIGINAL_CODE_ROOT"), "Original sources are optional")
class OriginalEquivalenceTests(unittest.TestCase):
    def test_extracted_numerical_functions(self):
        source_root = Path(os.environ["ORIGINAL_CODE_ROOT"])
        records = json.loads((ROOT / "source_map.json").read_text())
        originals = {}
        for record in records:
            path = source_root / record["source"]
            raw = path.read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), record["sha256"])
            tree = ast.parse(raw)
            functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                         and node.name in record["functions"]]
            namespace = {"np": np, "K": 200}
            if "v150_" in record["source"]:
                namespace["v148"] = types.SimpleNamespace(split=originals["split"])
            exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
            originals.update({node.name: namespace[node.name] for node in functions})
        data = geometry()
        np.testing.assert_equal(common.auroc(np.array([0., 2.]), np.array([0., 1.])),
                                originals["auc"](np.array([0., 2.]), np.array([0., 1.])))
        def equal(a, b):
            if isinstance(a, dict):
                self.assertEqual(a.keys(), b.keys())
                for key in a:
                    equal(a[key], b[key])
            elif isinstance(a, (tuple, list)):
                for left, right in zip(a, b):
                    equal(left, right)
            else:
                np.testing.assert_allclose(a, b, rtol=0, atol=0, equal_nan=True)
        cases = [
            (restore_activations.restoration, (data["pre"], data["thresholds"])),
            (restore_activations.gain_formula, (np.ones((24, 3)), data["pre"], np.ones((2, 3)))),
            (decompose_loss.statistics, (np.array([[0.8, 0.9]]), np.array([0.8]), np.array([0.6]), np.array([0.55]))),
            (decompose_loss.interval, (0.7, np.array([0.6, 0.7, 0.8]))),
            (rotate_encoder.reference_fraction, (data["eigenvalues"], data["basis"], 1, 123)),
            (rotate_encoder.matched_edit, (data["beta"], data["basis"], data["reference"], data["angle"], 1)),
            (rotate_encoder.control, (data["beta"], data["feature_ids"], data["angle"], 14801)),
            (edit_components.components, (data["beta"], data["eigenvalues"], data["angle"], 1)),
            (edit_components.matched_random, (data["beta"], data["beta"] * 0.5, data["feature_ids"], 15901)),
            (detect_entities.auc_bank, (data["pre"], np.eye(2)[data["labels"]])),
            (detect_entities.cutoff, (data["pre"][:, 0], data["labels"] == 0)),
            (detect_entities.counts, (data["labels"], (data["pre"][:, 0] > 0).astype(int))),
            (retrieve_sentences.ranking, (data["pre"][:, 0], np.arange(24))),
            (retrieve_sentences.retrieval, (np.arange(24.), data["labels"].astype(bool), np.arange(23, -1, -1), 3)),
            (retrieve_sentences.bootstrap_tp, (np.arange(24), data["labels"].astype(bool), np.ones((3, 24), int), 3)),
        ]
        for function, arguments in cases:
            with self.subTest(function=function.__name__):
                equal(function(*arguments), originals[function.__name__](*arguments))


if __name__ == "__main__":
    unittest.main()
