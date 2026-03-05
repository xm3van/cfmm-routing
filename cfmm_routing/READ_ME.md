```
cfmm-routing/
  pyproject.toml
  README.md

  src/
    market.py        # pool specs + pool math (all types) + graph helpers
    routing.py       # convex model build/solve + flow extraction
    experiments.py   # experiment runner + caching 
    metrics.py       # slippage curves + summaries (HHI, pool counts, etc.)
    plots.py         # plotting functions (slippage, composition grid, dists)
    config.py        # dataclasses + validation + defaults
```

Research questions: 

Base experiments 
- distribution of liquidity 
- n_swap path
- pool diversity 
- global liquidity 

**Lending**
1) Liquidation capacity curve (core risk primitive)
  - Question: For a given collateral C and debt asset D, what’s the max size Q you can liquidate while staying under slippage cap s_max?
  - Design: Fix market graph (pools), treat liquidation as trade C→D. Sweep dx.

2) Liquidation under stressed liquidity (haircut to liquidity)
  - Question: How fragile is liquidation to liquidity withdrawals / market stress?
  - Design: Multiply all pool liquidities by stress factor g∈{1,0.7,0.5,0.3} (or remove top pools).

3) Concentration / single-point-of-failure for liquidations
  - Question: Are liquidations relying on one pool/venue too much?

4) Two-phase liquidation strategy: direct vs multi-hop
  - Question: When should the liquidator use direct pool vs route via intermediates?

5) Liquidation bonus sufficiency test (economic viability) 
  - Question: Is liquidation bonus high enough to compensate slippage + fees + hop-length?

**DEX**
1) “Add-one-edge” pool placement (the MVP network design problem)
  - Question: If you can add one new pool, which pair (i,j) gives best improvement?
2) Budget allocation across multiple new pools (top-k selection)
  - Question: With budget B, should you add one big pool or several smaller ones, and where?
3) Bridge vs deepen: is it better to connect components or deepen existing routes?


Practical application - Optimization of the liquidity Network:  
- Product for Dexes to identify promising new pool deployments that overall improve routing efficient 
- Product for lending protocols to incentivise pool most improving liquidation efficiency for set paths



### Next steps 

#### 15-01-2026
- Trade Size volume
  - Comparing numerically 2 or 3 curves at certain levels 
  - Sensitvity analysis of how much does it change based on composition 

  - Trade-off: 
      - Budget for liquidity vs routing

- Curve data on pools 
  - curve.prices.api --> Not viable 

- Meeting for project description: 
  - What go going on? 
  - Prepare overleaf
    - go through the research process 
    - formalise problem 

- Invite supervisor: (after meet)
  - Alain invite co-author
  - Write an email 

#### 22-01-2026
How about this claim: 
- What mimimum parameter set do we need to aggreate cost curve?
  - Nodes = Tokens
  - Link = Liquidity Pools
  - Directional = Trade (A-->B)
  - Sparse & Small world network 

- Empirical Sample (Liquidty Pools Tokens) 1000 
  - Major DEXes, Top DEXes: 
    - Uniswap 
    - Curve 
    - 

  - Diameter
  - Distribution 

- Invite supervisor: (after meet)
  - Alain invite co-author
  - Write an email 


#### 29-01-2026
- Fromalise Objective Function 
  - check docuemntation aggreators 
- Fromalise generative model 
    - More Curve Snapshots (?) 
    - Idea:
      - Curve network as base (?) 

      