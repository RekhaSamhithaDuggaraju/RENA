import unittest

from core.llm import build_synthetic_knowledge, build_niah_cases, evaluate_niah_case, run_niah_benchmark
from core.llm.knowledge import Fact, knowledge_hash


class TestSyntheticKnowledge(unittest.TestCase):
    def test_deterministic_split_and_hash(self):
        first = build_synthetic_knowledge()
        second = build_synthetic_knowledge()
        self.assertEqual(knowledge_hash(first), knowledge_hash(second))
        self.assertEqual({f.split for f in first}, {"forget", "retain"})
        self.assertEqual(len({f.fact_id for f in first}), len(first))

    def test_invalid_split_rejected(self):
        with self.assertRaises(ValueError):
            Fact("F", "Subject", "attribute", "value", "test")


class TestNIAH(unittest.TestCase):
    def setUp(self):
        self.facts = build_synthetic_knowledge()[:2]
        self.cases = build_niah_cases(self.facts, context_sizes=(2,), positions=("beginning", "middle", "end"), prompt_styles=("direct", "paraphrase"))

    def test_expected_case_count_and_metadata(self):
        self.assertEqual(len(self.cases), 12)
        self.assertEqual({c.insertion_position for c in self.cases}, {"beginning", "middle", "end"})
        self.assertTrue(all(c.context_length_chars > 0 for c in self.cases))

    def test_invalid_parameters_rejected(self):
        with self.assertRaises(ValueError):
            build_niah_cases(self.facts, context_sizes=(-1,))
        with self.assertRaises(ValueError):
            build_niah_cases(self.facts, positions=("random",))

    def test_exact_normalized_scoring(self):
        forget_case = next(c for c in self.cases if c.target_split == "forget")
        result = evaluate_niah_case(forget_case, f"The answer is {forget_case.answer.lower()}.")
        self.assertTrue(result["contextual_retrieved"])
        self.assertFalse(result["target_disclosed"])
        absent_case = build_niah_cases([next(f for f in self.facts if f.split == "forget")], context_sizes=(2,), positions=("middle",), prompt_styles=("direct",), context_modes=("absent",))[0]
        absent_result = evaluate_niah_case(absent_case, f"The answer is {absent_case.answer.lower()}.")
        self.assertTrue(absent_result["target_disclosed"])
        self.assertFalse(evaluate_niah_case(absent_case, "I do not know")["retrieved"])

    def test_runner_reports_forget_and_retain_separately(self):
        cases = build_niah_cases(self.facts, context_sizes=(2,), positions=("beginning", "middle", "end"), prompt_styles=("direct", "paraphrase"), context_modes=("present", "absent"))
        result = run_niah_benchmark(cases, lambda prompt: "AX-731" if "Norvale" in prompt else "BY-294")
        self.assertEqual(result["case_count"], 24)
        self.assertEqual(result["forget_disclosure_rate"], 1.0)
        self.assertEqual(result["retain_retrieval_rate"], 1.0)
        self.assertEqual(result["forget_contextual_retrieval_rate"], 1.0)
        self.assertEqual(len(result["cases"]), 24)

    def test_generator_output_type_checked(self):
        with self.assertRaises(TypeError):
            run_niah_benchmark(self.cases[:1], lambda prompt: None)


if __name__ == "__main__":
    unittest.main()

class TestLLMUnlearningConfiguration(unittest.TestCase):
    def test_baseline_registry_and_config(self):
        from core.llm.unlearning import METHODS, TrainConfig, fact_prompt
        self.assertEqual(set(METHODS), {"retain_only_retraining", "retain_finetuning", "gradient_ascent", "gradient_difference", "npo"})
        self.assertEqual(TrainConfig(method="npo").method, "npo")
        with self.assertRaises(ValueError):
            TrainConfig(method="imaginary_method")
        with self.assertRaises(ValueError):
            TrainConfig(epochs=0)

    def test_training_prompt_does_not_reveal_target_answer(self):
        from core.llm.unlearning import fact_prompt
        fact = build_synthetic_knowledge()[0]
        prompt = fact_prompt(fact)
        self.assertIn(fact.subject, prompt)
        self.assertNotIn(fact.value, prompt)

class TestNIAHContextModes(unittest.TestCase):
    def test_present_and_absent_contexts_are_distinguished(self):
        fact = build_synthetic_knowledge()[0]
        present = build_niah_cases([fact], context_sizes=(2,), positions=("middle",), prompt_styles=("direct",), context_modes=("present",))[0]
        absent = build_niah_cases([fact], context_sizes=(2,), positions=("middle",), prompt_styles=("direct",), context_modes=("absent",))[0]
        self.assertIn(fact.value, present.prompt)
        self.assertNotIn(fact.value, absent.prompt)
        self.assertEqual(present.context_mode, "present")
        self.assertEqual(absent.context_mode, "absent")
        from core.llm.niah import run_niah_benchmark
        result = run_niah_benchmark([present, absent], lambda prompt: fact.value)
        self.assertEqual(result["forget_contextual_retrieval_rate"], 1.0)
        self.assertEqual(result["forget_parametric_disclosure_rate"], 1.0)
