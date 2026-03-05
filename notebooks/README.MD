# Working Notes: 

**Modularity**
```
SBM config
    ↓
SBMGenerator
    ↓
Graph
    ↓
GraphToMarketConverter
    ↓
MarketConfig
    ↓
run_sweep
```

---

# I. Research Objective

1. What is the primary purpose of the SBM?
A.: We are trying to study the shape of the aggregated cost function across different tradesizes after routing through the network. We are trying to understand how the function shape is impacted by it composition. The goal of the SBM is to generate different network typologies (tuneable to empirical data) but generally detached. 

2. Are synthetic markets meant to approximate real DeFi data? (Yes/No)
No, not necessarily. 

3. What routing metric are you ultimately studying?
the variation of the shape of the aggregated cost function across different tradesizes after routing through the network
---

# II. Structural Topology

4. How many structural roles?
Let begin with (core/mid/periphery)

5. Should the graph always be connected? (Yes / Drop LCC / Resample)
It doesn't have to but we will always route from a given asset (e.g. A) to another (e.g. B)

6. Use degree correction (heavy-tailed hubs)? (Yes/No)
Yes, but we can introdcue it later

7. Should role connectivity matrix be symmetric? (Yes/No)
Yes

8. Will you vary:

   * role proportions?
   * connectivity intensity?
   * both?

---

# III. Token Typology

9. Is token_type conditional on role? (Yes/No)

10. Are token types:

    * Stable
    * Major
    * Alt
    * Meme
      (Confirm or redefine set)

11. Should token_type influence:

    * AMM choice?
    * Liquidity?
    * Fees?
    * All?
    * None?

---

# IV. Pool Structure

12. Pools per edge:

    * 1 always
    * Fixed k
    * Sample k
    * Conditional on role pair

13. Should core-core edges have more pools? (Yes/No)

14. Can multiple pools on same edge share the same AMM type? (Yes/No)

---

# V. AMM Assignment

15. Is AMM distribution conditional on:

    * role pair?
    * token types?
    * both?
    * neither?

16. Pool types allowed:

    * univ2
    * bal_wgm
    * curve
      (confirm or modify)

17. Should AMM params (A, weights, etc.) be:

    * fixed constants
    * sampled
    * role dependent?

---

# VI. Liquidity Model

18. Total liquidity per edge depends on:

    * role pair?
    * token types?
    * global constant?
    * sampled heavy-tailed?

19. How is liquidity split across multiple pools?

    * equal
    * Dirichlet
    * AMM-type weighted
    * deterministic rule

20. Should global liquidity scale with number of nodes? (Yes/No)

---

# VII. Fees

21. Fee structure:

    * global constant
    * AMM dependent
    * role dependent
    * token dependent

---

# VIII. Experiment Design

22. Will you:

    * generate one graph per experiment
    * Monte Carlo average over many graphs

23. Which structural parameters will you vary systematically?

24. Do you need reproducibility across runs? (Yes/No)

---

# IX. Complexity Control

25. What must remain fixed across experiments?

    * AMM params?
    * liquidity distribution?
    * fee?
    * pool multiplicity?

26. Maximum number of free parameters you want in SBM? (roughly)

---

# X. Integration Policy

27. Should SBM output:

    * only MarketConfig
    * also diagnostics (degree stats etc.)

28. Should synthetic experiments live:

    * separate exp files
    * integrated into existing ones

---

Answer these concisely.
Once you paste answers, I will collapse everything into a clean, coherent SBM blueprint tailored to your routing research.
