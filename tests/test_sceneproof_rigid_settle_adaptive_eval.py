import unittest
import csv
import json
import tempfile
from argparse import Namespace
from pathlib import Path

from sceneproof_rigid_settle_adaptive_eval_fix84 import compose, probe_route


class RigidSettleAdaptiveRouteTest(unittest.TestCase):
    def probe(self, stability="unstable", **updates):
        row = {
            "status": "measured",
            "incumbent_restored": True,
            "new_collision_object_ids": [],
            "after_support": {
                "certificate_status": "certified",
                "stability_class": stability,
                "declared_parent_contact_present": True,
            },
        }
        row.update(updates)
        return row

    def test_stable_probe_is_selected_without_retry(self):
        self.assertEqual(probe_route(self.probe("stable"), "primary"), "select")

    def test_unstable_probe_retries_only_in_order(self):
        probe = self.probe("unstable")
        self.assertEqual(probe_route(probe, "primary"), "retry_damping")
        self.assertEqual(probe_route(probe, "damping"), "retry_friction")
        self.assertEqual(probe_route(probe, "friction"), "reject")

    def test_new_collision_fails_closed(self):
        self.assertEqual(
            probe_route(
                self.probe("stable", new_collision_object_ids=["desk_0"]),
                "primary",
            ),
            "reject",
        )

    def test_unproven_support_does_not_trigger_parameter_search(self):
        probe = self.probe("stable")
        probe["after_support"]["certificate_status"] = "abstained"
        self.assertEqual(probe_route(probe, "primary"), "reject")

    def test_restoration_failure_fails_closed(self):
        self.assertEqual(
            probe_route(self.probe("stable", incumbent_restored=False), "primary"),
            "reject",
        )

    def test_blender_probe_forwards_explicit_passive_and_world_settings(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "modules"
            / "S4_blender_layout_and_corr.py"
        ).read_text(encoding="utf-8")
        self.assertIn("IMAGINARIUM_SETTLE_PASSIVE_FRICTION", source)
        self.assertIn("IMAGINARIUM_SETTLE_SUBSTEPS", source)
        self.assertIn("IMAGINARIUM_SETTLE_SOLVER_ITERATIONS", source)
        self.assertIn("passive_settings=passive_settings", source)
        self.assertIn("world_settings=world_settings", source)

    def test_compose_retains_multiple_accepted_objects_in_one_scene(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene = "official_01"
            baseline = "baseline"
            target = "target"
            source = root / f"{scene}_{baseline}_result" / "S4_layout_refinement"
            source.mkdir(parents=True)
            (source / f"{scene}_{baseline}_placement_info_s4.json").write_text(
                json.dumps({"obj_info": {"a": {}, "b": {}}}), encoding="utf-8"
            )
            probes = []
            for object_id, x in (("a", 1.0), ("b", 2.0)):
                probe = root / f"{object_id}.json"
                probe.write_text(
                    json.dumps({"settled_pose_matrix": [[1, 0, 0, x]]}),
                    encoding="utf-8",
                )
                probes.append((object_id, probe))
            selected = root / "selected.tsv"
            with selected.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["scene", "object_id", "accepted", "probe"],
                    delimiter="\t",
                )
                writer.writeheader()
                for object_id, probe in probes:
                    writer.writerow(
                        {"scene": scene, "object_id": object_id, "accepted": "true", "probe": probe}
                    )
            manifest = root / "manifest.txt"
            manifest.write_text(scene + "\n", encoding="utf-8")
            certificate = root / "certificate.json"
            compose(
                Namespace(
                    selected=selected,
                    manifest=manifest,
                    saved_results=root,
                    baseline_version=baseline,
                    target_version=target,
                    target_manifest=root / "target_manifest.txt",
                    certificate=certificate,
                )
            )
            audit = json.loads(certificate.read_text(encoding="utf-8"))
            self.assertEqual(
                audit["scenes"][scene]["retained_changed_objects"], ["a", "b"]
            )


if __name__ == "__main__":
    unittest.main()
