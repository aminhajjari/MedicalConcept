
import sys
sys.path.append('.')

from src.models.Mistral import Mistral
from typing import Dict
import re

# ── local checkpoint path ──────────────────────────────────────────────────────
MISTRAL_LOCAL_CKPT = (
    "/home/gkianfar/scratch/Amin/concept/maincode/"
    "Self-2-step-concept-based-skin-diagnosis/checkpoint/Mistral-7B-Instruct"
)


class MistralBasedRefiner:
    """
    Mistral-7B-Instruct based refiner.
    Identical interface to MMedBasedRefiner so it is a drop-in replacement.
    """

    def __init__(self, ckpt: str = MISTRAL_LOCAL_CKPT):
        print(f"Loading Mistral refiner from: {ckpt}")
        self.model = Mistral(ckpt=ckpt)
        print("✓ Mistral refiner loaded")

    # ── public call ─────────────────────────────────────────────────────────────
    def __call__(self, concepts_str: str, feedback: str,
                 concepts_dict: Dict[str, str]) -> str:

        instruction = self._build_instruction()
        # Seed the answer with "The color is" so Mistral continues inline
        query = self._build_query(concepts_str, feedback)

        try:
            prompt = self.model.get_prompt(
                instruction=instruction, query=query, demos=None
            )
            raw = self.model.predict(prompt=prompt, max_new_tokens=120).strip()
            cleaned = self._clean_output(raw)

            if self._validate_format(cleaned):
                print("✓ Mistral refinement successful")
                return cleaned

            print("⚠ Mistral format failed, attempting extraction...")
            extracted = self._try_extract_concepts(cleaned, concepts_dict)
            if extracted and self._validate_format(extracted):
                print("✓ Extraction successful")
                return extracted

            print("⚠ Mistral produced invalid output — keeping ORIGINAL concepts (no-op)")
            return concepts_str

        except Exception as e:
            import traceback
            print(f"⚠ Mistral error [{type(e).__name__}]: {repr(e)} — keeping ORIGINAL concepts (no-op)")
            traceback.print_exc()
            return concepts_str

    # ── prompt builders ──────────────────────────────────────────────────────────
    def _build_instruction(self) -> str:
        return (
            "You are a dermatology expert. Fix ONLY the clinically inconsistent "
            "concepts listed in VIOLATIONS. "
            "Output ONLY the corrected full description — no explanations, "
            "no preamble, no bullet points. "
            "Use this EXACT format: "
            "The color is ..., the shape is ..., the border is ..., "
            "the dermoscopic patterns are ..., the texture is ..., "
            "the symmetry is ..., the elevation is ..."
        )

    def _build_query(self, concepts_str: str, feedback: str) -> str:
        violation_list = "\n".join(
            f"• {v}" for v in feedback.split("\n") if v.strip()
        )
        return (
            f"CURRENT DESCRIPTION:\n{concepts_str}\n\n"
            f"VIOLATIONS TO FIX:\n{violation_list}\n\n"
            f"CORRECTED DESCRIPTION (start immediately with 'The color is'):\n"
            f"The color is"
        )

    # ── output cleaning ──────────────────────────────────────────────────────────
    def _clean_output(self, output: str) -> str:
        # The prompt seed "The color is" may have been stripped by predict()
        # Try to find a complete concept sentence first
        match = re.search(
            r'The color is .+?elevation is [^.]+\.', output,
            re.IGNORECASE | re.DOTALL
        )
        if match:
            return match.group(0)

        # predict() strips the prompt, so output begins mid-sentence after seed
        # Re-attach the seed if needed
        stripped = output.split("\n")[0].strip()
        if stripped and not stripped.lower().startswith("the color is"):
            stripped = "The color is " + stripped

        # Remove common preambles Mistral sometimes adds
        stripped = re.sub(
            r'^(here is|refined description|corrected description)[\s:]*',
            '', stripped, flags=re.IGNORECASE
        )
        return stripped.strip()

    # ── validation ───────────────────────────────────────────────────────────────
    def _validate_format(self, s: str) -> bool:
        keywords = ['color', 'shape', 'border', 'pattern',
                    'texture', 'symmetry', 'elevation']
        return sum(1 for k in keywords if k in s.lower()) >= 5

    # ── concept salvage ──────────────────────────────────────────────────────────
    def _try_extract_concepts(self, output: str,
                               original_dict: Dict[str, str]) -> str:
        extracted = {}
        keys = ['color', 'shape', 'border', 'dermoscopic patterns',
                'texture', 'symmetry', 'elevation']

        for key in keys:
            patterns = [
                rf"(?:the\s+)?{re.escape(key)}\s+(?:is|are)\s+([^,\.]+)",
                rf"{re.escape(key)}:\s*([^,\.]+)",
            ]
            for pat in patterns:
                m = re.search(pat, output, re.IGNORECASE)
                if m:
                    extracted[key] = m.group(1).strip()
                    break
            if key not in extracted and key in original_dict:
                extracted[key] = original_dict[key]

        if len(extracted) >= 5:
            template = (
                "The color is {color}, the shape is {shape}, "
                "the border is {border}, "
                "the dermoscopic patterns are {dermoscopic patterns}, "
                "the texture is {texture}, "
                "the symmetry is {symmetry}, the elevation is {elevation}."
            )
            try:
                return template.format(**extracted)
            except KeyError:
                return ""
        return ""
