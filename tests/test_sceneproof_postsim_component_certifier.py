import unittest

from sceneproof_postsim_component_certifier import (
    family_gates,
    relation_neighborhoods,
    scoped_component,
)


class SceneProofPostsimComponentCertifierTest(unittest.TestCase):
    def test_family_gate_never_allows_macro_to_hide_support(self):
        incumbent = {
            "headline_macro_realizability": 0.5,
            "families": {
                name: {"score": 0.5}
                for name in ("collision", "support", "plane", "semantic")
            },
        }
        candidate = {
            "headline_macro_realizability": 0.6,
            "families": {
                "collision": {"score": 0.8},
                "support": {"score": 0.49},
                "plane": {"score": 0.55},
                "semantic": {"score": 0.56},
            },
        }
        passed, gates = family_gates(incumbent, candidate, 0.005)
        self.assertFalse(passed)
        self.assertFalse(gates["support"]["passed"])

    def test_floor_does_not_join_unrelated_support_components(self):
        graph = relation_neighborhoods(
            {
                "cup_0": {"supported": "table_0"},
                "table_0": {"supported": "floor_0"},
                "chair_0": {"supported": "floor_0"},
                "floor_0": {},
            }
        )
        self.assertEqual(graph["table_0"], {"cup_0"})
        self.assertEqual(graph["chair_0"], set())

    def test_scope_includes_changed_chain_and_separator_only(self):
        graph = {
            "cup_0": {"table_0"},
            "table_0": {"cup_0", "book_0"},
            "book_0": {"table_0"},
        }
        scope = scoped_component(
            "cup_0", graph, {"cup_0", "table_0"}
        )
        self.assertEqual(scope, {"cup_0", "table_0", "book_0"})


if __name__ == "__main__":
    unittest.main()
