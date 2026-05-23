"""HPO module for Kronos-R.

Phase 1:  Bits-per-quantizer search  →  python -m hpo.phase1_bits_search
Phase 1s: Bits 3–5 supplement          →  python -m hpo.phase1_sup
Phase 2:  Tokenizer HPO               →  python -m hpo.phase2_tokenizer
Phase 3:  BaseModel HPO               →  planned

Evaluation:
  Phase 1 downstream  →  python -m hpo.evaluate_phase1_bits
  Phase 2 downstream  →  python -m hpo.evaluate_phase2
  Plots               →  python -m hpo.plot_phase1
"""
