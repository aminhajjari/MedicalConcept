
import sys
sys.path.append('.')

from src.models.MMed_Llama_3_8B import MMedLlama3
from typing import Dict
import re


class MMedBasedRefiner:
    """
    MMed-LLM based refiner that uses MMed-Llama-3-8B to refine concept predictions.
    """

    def __init__(self, ckpt="/home/gkianfar/scratch/Amin/concept/maincode/Self-2-step-concept-based-skin-diagnosis/checkpoint/MMed-Llama-3-8B-EnIns"):
        print(f"Loading MMed-LLM refiner: {ckpt}")
        self.model = MMedLlama3(ckpt=ckpt)
        print("✓ MMed-LLM refiner loaded")

    def __call__(self, concepts_str: str, feedback: str, concepts_dict: Dict[str, str]) -> str:
        violated_concepts = self._extract_violated_concepts(feedback)
        instruction = self._build_instruction(violated_concepts)
        query = self._build_query(concepts_str, feedback, violated_concepts)

        try:
            prompt = self.model.get_prompt(instruction=instruction, query=query, demos=None)
            refined_concepts = self.model.predict(prompt=prompt, max_new_tokens=500).strip()
            refined_concepts = self._clean_output(refined_concepts)

            if self._validate_format(refined_concepts):
                from src.self_refiner.concept_refiner import (
                    ConceptConsistencyRules, ConceptSelfRefine, SimpleRuleBasedRefiner)
                _parser = ConceptSelfRefine(llm_refine_fn=None)
                _rules  = ConceptConsistencyRules()
                before = len(_rules.check_consistency(concepts_dict))
                after  = len(_rules.check_consistency(_parser.parse_concepts(refined_concepts)))
                if after < before:
                    print(f"✓ LLM refinement successful ({before}->{after} violations)")
                    return refined_concepts
                else:
                    print(f"⚠ LLM echoed input ({before}->{after}), using rule-based fix")
                    return SimpleRuleBasedRefiner()(concepts_str, feedback, concepts_dict)
            else:
                print(f"⚠ Format validation failed, attempting extraction...")
                extracted = self._try_extract_concepts(refined_concepts, concepts_dict)

                if extracted and self._validate_format(extracted):
                    print("⚠ LLM produced invalid output — using extracted concepts")
                    return extracted
                else:
                    print("⚠ LLM produced invalid output — keeping ORIGINAL concepts (no-op)")
                    return concepts_str
        except Exception as e:
            import traceback
            print(f"⚠ LLM refinement error [{type(e).__name__}]: {repr(e)} — keeping ORIGINAL concepts (no-op)")
            return concepts_str

    def _extract_violated_concepts(self, feedback: str) -> set:
        violated = set()
        feedback_lower = feedback.lower()

        concept_keywords = {
            'border': ['border'],
            'color': ['color', 'multiple colors'],
            'shape': ['shape'],
            'symmetry': ['symmetry', 'asymmetric', 'symmetrical'],
            'texture': ['texture', 'smooth', 'ulcerated'],
            'elevation': ['elevation', 'flat', 'raised'],
            'dermoscopic patterns': ['pattern', 'network', 'veil', 'streak']
        }

        for concept, keywords in concept_keywords.items():
            if any(kw in feedback_lower for kw in keywords):
                violated.add(concept)

        return violated

    def _build_instruction(self, violated_concepts: set) -> str:
        base_instruction = """You are a dermatology expert. Fix clinical inconsistencies in dermoscopic descriptions.

CRITICAL RULES:
1. Output ONLY the complete refined description (no preambles, no explanations)
2. Use this EXACT format: "The color is ..., the shape is ..., the border is ..., the dermoscopic patterns are ..., the texture is ..., the symmetry is ..., the elevation is ..."
3. ONLY modify concepts that are clinically inconsistent
4. Keep all other concepts EXACTLY as they are

CLINICAL CONSISTENCY RULES:"""

        if 'border' in violated_concepts or 'symmetry' in violated_concepts:
            base_instruction += """
- Asymmetric lesions → Use "often blurry and irregular" border
- Symmetric lesions → Use "sharp and well-defined" border"""

        if 'color' in violated_concepts or 'dermoscopic patterns' in violated_concepts:
            base_instruction += """
- Multiple colors → Use "atypical pigment network, irregular streaks" patterns
- Single color → Use "regular pigment network, symmetric dots" patterns"""

        if 'texture' in violated_concepts or 'elevation' in violated_concepts:
            base_instruction += """
- Smooth texture → Must be "flat to slightly raised" elevation
- Raised/ulcerated → Cannot have "smooth" texture"""

        base_instruction += """

EXAMPLES OF GOOD FIXES:
❌ BAD: "The color is multiple colors, border is sharp, symmetry is asymmetric"
✅ GOOD: "The color is multiple colors, border is often blurry and irregular, symmetry is asymmetric"

❌ BAD: "The texture is smooth, elevation is raised with possible ulceration"
✅ GOOD: "The texture is smooth, elevation is flat to slightly raised"
"""
        return base_instruction

    def _build_query(self, concepts_str: str, feedback: str, violated_concepts: set) -> str:
        """Build query with seeded output to force MMed to continue writing."""

        violation_list = "\n".join([f"• {v}" for v in feedback.split('\n') if v])

        query = (
            f"CURRENT DESCRIPTION:\n{concepts_str}\n\n"
            f"VIOLATIONS TO FIX:\n{violation_list}\n\n"
            f"TASK: Rewrite fixing ONLY the violated concepts.\n"
            f"The color is"
        )

        return query

    def _clean_output(self, output: str) -> str:
        """Clean MMed output to extract just the concept description."""

        # Remove common preambles
        output = re.sub(
            r'^(refined description|here is|the refined|your refined description)[\s:]*',
            '', output, flags=re.IGNORECASE
        )

        # Remove bullet points
        output = re.sub(r'^\s*[-•*]\s*', '', output, flags=re.MULTILINE)

        # predict() strips the prompt seed "The color is", so reattach it
        if not output.lower().startswith('the color is'):
            output = 'The color is ' + output

        # Extract first complete concept sentence
        match = re.search(
            r'The color is .+?elevation is [^.]+\.', output,
            re.IGNORECASE | re.DOTALL
        )
        if match:
            return match.group(0)

        # Fallback: find line starting with "The color is"
        for line in output.split('\n'):
            line = line.strip()
            if line.lower().startswith('the color is'):
                return line

        # Fallback: find any line with 4+ concept keywords
        for line in output.split('\n'):
            line = line.strip()
            if len(line) > 50:
                keyword_count = sum(
                    1 for kw in ['color', 'shape', 'border', 'pattern', 'texture', 'symmetry', 'elevation']
                    if kw in line.lower()
                )
                if keyword_count >= 4:
                    return line

        # Last resort
        for line in output.split('\n'):
            if len(line.strip()) > 50:
                return line.strip()

        return output.strip()

    def _validate_format(self, refined_str: str) -> bool:
        """Check if at least 5/7 concept keywords are present."""
        concept_keywords = ['color', 'shape', 'border', 'pattern', 'texture', 'symmetry', 'elevation']
        found_count = sum(1 for keyword in concept_keywords if keyword in refined_str.lower())
        return found_count >= 5

    def _try_extract_concepts(self, output: str, original_dict: Dict[str, str]) -> str:
        """Try to salvage partially valid output by extracting concepts."""
        extracted_dict = {}
        concept_keys = ['color', 'shape', 'border', 'dermoscopic patterns',
                        'texture', 'symmetry', 'elevation']

        for key in concept_keys:
            patterns = [
                rf"(?:the\s+)?{re.escape(key)}\s+(?:is|are)\s+([^,\.]+)",
                rf"{re.escape(key)}:\s*([^,\.]+)",
            ]
            found = False
            for pattern in patterns:
                match = re.search(pattern, output, re.IGNORECASE)
                if match:
                    extracted_dict[key] = match.group(1).strip()
                    found = True
                    break
            if not found and key in original_dict:
                extracted_dict[key] = original_dict[key]

        if len(extracted_dict) >= 5:
            template = (
                "The color is {color}, the shape is {shape}, the border is {border}, "
                "the dermoscopic patterns are {dermoscopic patterns}, the texture is {texture}, "
                "the symmetry is {symmetry}, the elevation is {elevation}."
            )
            try:
                return template.format(**extracted_dict)
            except KeyError:
                return ""

        return ""


if __name__ == "__main__":
    print("Testing MMed-LLM Refiner (Enhanced Version)")
    print("=" * 80)

    refiner = MMedBasedRefiner(
        ckpt="/home/gkianfar/scratch/Amin/concept/maincode/Self-2-step-concept-based-skin-diagnosis/checkpoint/MMed-Llama-3-8B-EnIns"
    )

    test_concepts = (
        "The color is highly variable, often with multiple colors (black, brown, red, white, blue), "
        "the shape is irregular, "
        "the border is sharp and well-defined, "
        "the dermoscopic patterns are regular pigment network, symmetric dots and globules, "
        "the texture is smooth, "
        "the symmetry is asymmetrical, "
        "the elevation is flat to raised."
    )

    test_feedback = (
        "Asymmetric lesions need irregular borders (not sharp/well-defined)\n"
        "Multiple colors need irregular patterns (not regular)"
    )

    test_dict = {
        'color': 'highly variable, often with multiple colors (black, brown, red, white, blue)',
        'shape': 'irregular',
        'border': 'sharp and well-defined',
        'dermoscopic patterns': 'regular pigment network, symmetric dots and globules',
        'texture': 'smooth',
        'symmetry': 'asymmetrical',
        'elevation': 'flat to raised'
    }

    print("\nOriginal:")
    print(test_concepts)
    print("\nFeedback:")
    print(test_feedback)

    print("\n--- Testing Refinement ---")
    refined = refiner(test_concepts, test_feedback, test_dict)
    print("Refined:")
    print(refined)

    print("\n✓ Test complete!")
