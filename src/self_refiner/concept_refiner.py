import re
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConceptConsistencyRules:
    """Medical consistency rules for dermoscopic concepts."""
    
    @staticmethod
    def check_consistency(concepts_dict: Dict[str, str]) -> List[str]:
        """Check for clinical inconsistencies in concept predictions."""
        violations = []

        # Rule 1: Asymmetry + Border
        if 'asymmetric' in concepts_dict.get('symmetry', '').lower():
            border = concepts_dict.get('border', '').lower()
            if 'sharp' in border or 'well-defined' in border:
                violations.append(
                    "Asymmetric lesions need irregular borders (not sharp/well-defined)"
                )

        
        # Rule 2: Multiple Colors + Patterns
        if 'multiple colors' in concepts_dict.get('color', '').lower():
            patterns = concepts_dict.get('dermoscopic patterns', '').lower()
            if re.search(r'\bregular\b', patterns):          # was: 'regular' in patterns
                violations.append(
                    "Multiple colors need irregular patterns (not regular)"
                )

        # Rule 3: Texture + Elevation consistency
        texture = concepts_dict.get('texture', '').lower()
        elevation = concepts_dict.get('elevation', '').lower()
        
        elevation_is_problematic = (
            'ulcerat' in elevation or 'raised with possible' in elevation
        )
        if 'smooth' in texture and elevation_is_problematic:
            violations.append(
                "Smooth texture conflicts with ulcerated/raised features"
            )

        # Rule 4: Atypical patterns + Symmetry
        patterns = concepts_dict.get('dermoscopic patterns', '').lower()
        sym = concepts_dict.get('symmetry', '').lower()
        if ('atypical' in patterns or 'veil' in patterns):
            if 'symmetrical' in sym and 'asymmetric' not in sym and 'depending on type' not in sym:
                violations.append(
                    "Atypical patterns need asymmetry (not symmetrical)"
                )

      
        # Rule 5: Irregular shape + Symmetric → contradiction
        shape = concepts_dict.get('shape', '').lower()
        if shape.strip() == 'irregular':
            sym = concepts_dict.get('symmetry', '').lower()
            if 'symmetrical' in sym and 'asymmetric' not in sym and 'depending on type' not in sym:
                violations.append(
                    "Irregular shape typically means asymmetry (not symmetrical)"
                )

        # Rule 6: Atypical patterns + Regular border → contradiction
        if 'atypical' in concepts_dict.get('dermoscopic patterns', '').lower():
            if 'sharp' in concepts_dict.get('border', '').lower():
                violations.append(
                    "Atypical patterns conflict with sharp/well-defined border"
                )

        # Rule 7: Flat elevation + Ulcerated texture → contradiction  
        elev = concepts_dict.get('elevation', '').lower()
        if 'flat' in elev and 'raised' not in elev:
            if 'ulcerated' in concepts_dict.get('texture', '').lower():
                violations.append(
                "Flat elevation conflicts with ulcerated texture"
                )

        # Rule 8: Multiple colors + Symmetrical → contradiction
        if 'multiple colors' in concepts_dict.get('color', '').lower():
            if 'symmetrical' in concepts_dict.get('symmetry', '').lower():
                if 'asymmetric' not in concepts_dict.get('symmetry', '').lower():
                    if 'depending on type' not in concepts_dict.get('symmetry', '').lower():
                        violations.append(
                        "Multiple colors typically indicate asymmetrical lesions"
                        )
     
        return violations


class SimpleRuleBasedRefiner:
    """Simple rule-based concept refiner (fallback when LLM fails)."""
    
    def __call__(self, concepts_str: str, feedback: str, concepts_dict: Dict[str, str]) -> str:
        """Apply hardcoded fixes for common violations."""
        refined = concepts_dict.copy()

        # Fix 1: Asymmetry → Irregular border
        if 'asymmetric' in refined.get('symmetry', '').lower():
            if 'sharp' in refined.get('border', '').lower():
                refined['border'] = 'often blurry and irregular'

       
        # Fix 2: Multiple colors → Irregular patterns
        if 'multiple colors' in refined.get('color', '').lower():
            if re.search(r'\bregular\b', refined.get('dermoscopic patterns', '').lower()):
                refined['dermoscopic patterns'] = 'atypical pigment network, irregular streaks'

        # Fix 3: Smooth texture → Flat elevation
        if 'smooth' in refined.get('texture', '').lower():
            elev = refined.get('elevation', '').lower()
            if 'ulcerat' in elev or 'raised with possible' in elev:
                refined['elevation'] = 'flat to slightly raised'

        # Fix 4: Atypical patterns → Asymmetry
        patterns = refined.get('dermoscopic patterns', '').lower()
        if 'atypical' in patterns or 'veil' in patterns:
            if 'symmetrical' in refined.get('symmetry', '').lower():
                refined['symmetry'] = 'asymmetrical'

        
        # Fix 5: Irregular shape → Asymmetry
        if refined.get('shape', '').lower().strip() == 'irregular':  # exact match
            sym = refined.get('symmetry', '').lower()
            if 'symmetrical' in sym and 'asymmetric' not in sym:
                refined['symmetry'] = 'asymmetrical'

        # Fix 6: Atypical patterns → Irregular border
        if 'atypical' in refined.get('dermoscopic patterns', '').lower():
            if 'sharp' in refined.get('border', '').lower():
                refined['border'] = 'often blurry and irregular'

        # Fix 7: Flat elevation + Ulcerated texture → fix texture
        if 'flat' in refined.get('elevation', '').lower():
            if 'ulcerated' in refined.get('texture', '').lower():
                refined['texture'] = 'smooth'

        # Fix 8: Multiple colors → Asymmetry
        if 'multiple colors' in refined.get('color', '').lower():
            sym = refined.get('symmetry', '').lower()
            if 'symmetrical' in sym and 'asymmetric' not in sym:
                refined['symmetry'] = 'asymmetrical'

        
        # Convert back to string
        template = (
            "The color is {color}, the shape is {shape}, the border is {border}, "
            "the dermoscopic patterns are {dermoscopic patterns}, the texture is {texture}, "
            "the symmetry is {symmetry}, the elevation is {elevation}."
        )
        return template.format(**refined)


class ConceptSelfRefine:
    """
    Self-refine mechanism with improved oscillation handling.
    
    Key improvements:
    - Better stopping criteria
    - Graceful degradation when stuck
    - Accept "good enough" results (1 violation after 2 iterations)
    """
    
    def __init__(self, llm_refine_fn, max_iterations=3, verbose=True):
        self.llm_refine_fn = llm_refine_fn
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.rules = ConceptConsistencyRules()

    def parse_concepts(self, concepts_str: str) -> Dict[str, str]:
        """Parse concept string into dict; values may themselves contain commas."""
        concepts_dict = {}
        fields = ['color', 'shape', 'border', 'dermoscopic patterns',
                  'texture', 'symmetry', 'elevation']
        nxt = "|".join(re.escape(k) for k in fields)
        for key in fields:
            pattern = (
                rf"{re.escape(key)}\s+(?:is|are)\s+(.+?)"
                rf"(?:,\s*(?:the\s+)?(?:{nxt})\s+(?:is|are)\b|\.?\s*$)"
            )
            m = re.search(pattern, concepts_str, re.IGNORECASE | re.DOTALL)
            if m:
                concepts_dict[key] = m.group(1).strip().rstrip(',').strip()
        return concepts_dict

    def refine(self, initial_concepts: str, diagnosis: str = None,
               concept_margins: Dict = None, margin_threshold: float = 0.2) -> Tuple[str, Dict]:
        """
        Iteratively refine concepts.
        - Logical violations (from the rules) drive the loop and convergence.
        - Low-confidence concepts (small top1-top2 margin) are NOT counted as
          violations, but are passed to the LLM as a hint to reconsider.
        """
        current = initial_concepts
        history = [initial_concepts]
        violation_counts = []

        # Confidence signal: which concepts is the vision model unsure about?
        uncertain = []
        if concept_margins:
            uncertain = [k for k, m in concept_margins.items() if m < margin_threshold]
        uncertainty_note = ""
        if uncertain:
            uncertainty_note = (
                "The image model is low-confidence about: " + ", ".join(uncertain) +
                ". Reconsider these concepts so they are clinically consistent with the others."
            )
        
        info = {
            'iterations': 0,
            'initial_violations': 0,
            'final_violations': 0,
            'converged': False
        }
        
        for i in range(self.max_iterations):
            if self.verbose:
                print(f"\n--- Iteration {i} ---")
            
            # Check violations
            concepts_dict = self.parse_concepts(current)
            violations = self.rules.check_consistency(concepts_dict)
            n_viols = len(violations)
            violation_counts.append(n_viols)
            
            if i == 0:
                info['initial_violations'] = n_viols
            
            if self.verbose:
                print(f"Found {n_viols} consistency violations")
                if violations:
                    for v in violations:
                        print(f"  • {v}")
            
            
            # PERFECT: No violations. But give the LLM ONE pass if the vision
            # model flagged uncertain concepts on the first iteration.
            if n_viols == 0 and not (i == 0 and uncertain):
                if self.verbose:
                    print(f"✓ Perfect convergence at iteration {i}")
                info['converged'] = True
                info['iterations'] = i
                info['final_violations'] = 0
                break
            
            # GOOD ENOUGH: 1-2 violations after 2+ iterations
            if i >= 2 and n_viols <= 2:
                if self.verbose:
                    print(f"✓ Good enough: {n_viols} violations after {i} iterations")
                info['converged'] = True  # Mark as converged (good enough)
                info['iterations'] = i
                info['final_violations'] = n_viols
                break
            
            # STUCK: Same violations for 3 iterations
            if len(violation_counts) >= 3:
                last_three = violation_counts[-3:]
                if last_three[0] == last_three[1] == last_three[2]:
                    if self.verbose:
                        print(f"⚠ Stuck at {last_three[0]} violations for 3 iterations, stopping")
                    break
            
            
            # Try to refine
            feedback = "\n".join(violations)
            if i == 0 and uncertainty_note:
                feedback = (feedback + "\n" + uncertainty_note).strip()
            
            try:
                refined = self.llm_refine_fn(current, feedback, concepts_dict)

                if refined is None:
                    if self.verbose:
                        print(f"⚠ Refiner returned None, stopping")
                    break

                if refined == current:
                    if self.verbose:
                        print(f"⚠ No change from refiner, stopping")
                    break
                
                current = refined
                history.append(refined)
                
            except Exception as e:
                if self.verbose:
                    print(f"⚠ Refinement error: {e}, stopping")
                break
        
        info['iterations'] = i
        info['final_violations'] = n_viols
        
        return current, info
