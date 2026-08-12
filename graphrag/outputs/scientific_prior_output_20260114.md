## 1) Reframed context for the spatial evolution term (SE_i)

- Distance-based coupling is supported via empirical distance decay in connection probability, P(d) ~ d^(-1.5), suggesting kernels that down-weight long-range interactions [Data: Sources (3)].
- Gravity-type flows provide an explicit template linking node attributes and distances: T_ij ∝ (m_i m_j) / d_ij^D, with the travel-distance distribution P(d) ∝ d^(-D) in fractal spatial domains [Data: Sources (809)].
- Memory-driven mobility affects exploration vs. return dynamics, with the probability to choose a new location P_new = 1 / (1 + λ(ln S + C)) and rank-frequency f_r ∝ (λ S)/r + 1 − λ; higher λ concentrates movement among fewer locations, altering effective coupling patterns [Data: Sources (809)].
- Network flow assignment using cost-based routing (travel time vs. distance) and capacity limitation improves data fit; traffic values are computed on edges via equations (2–5), with travel time outperforming distance and capacity constraints yielding higher PCC (up to 0.752) [Data: Sources (279)].
- Flow generation via weighted betweenness centrality provides a principled network operator; the resulting traffic density is lognormal and tied to structural variability through a convolution over degree distribution and shell-size noise [Data: Sources (279)].
- Practical parameterization uses limited velocity classes (e.g., 90–40–15 mph) and iteration stopping rules (condition (5)), indicating implementable constraints for stable flow assignment [Data: Sources (279)].
- Empirical network overlap between proximity and communication layers is low (overall ~8.55%), motivating dual-network coupling with an overlap parameter Γ; efficacy of coupling (e.g., tracing) increases with Γ and the epidemic peak decays exponentially with Γ [Data: Sources (160)].
- Adaptive coupling via tracing effort reflects feedback: eff = Σ_{n∈(I∪S)} β_r + Σ_{n∈(Neighbour_t(T)∩(S∪I))} K_t(n) β_t; more infections induce higher tracing effort, explaining effectiveness even at low Γ [Data: Sources (162)].
- Community geography constrains spatial spread: geographic span D(s) and the empirical observation that real spans are much smaller than null spans (and cluster counts scale with size until a regime change) support structured, non-uniform spatial coupling [Data: Sources (3)].
- Simulation parameterizations for contagion on networks (β_r, β_t, γ, Δt, K, population size) provide concrete variable choices and timescale mappings when SE_i represents disease-related spatial coupling [Data: Sources (160)].

## 2) Cited raw excerpts (closest-to-source material)

- [E1] “D(s) = (1/n_s) ∑_{i∈C_s} sqrt((X_s − x_i)^2 + (Y_s − y_i)^2), (1) … large values of D indicate that the members of the community are geographically spread out.”  
  Why this matters for SE_i: Provides a formal measure of spatial dispersion that can inform normalized or scale-aware coupling operators [Data: Sources (3)].

- [E2] “The connection probability decays with distance as P(d) ∼ d^−1.5.”  
  Why this matters for SE_i: Justifies inverse power-law distance kernels for spatial coupling [Data: Sources (3)].

- [E3] “We found that the number of spatial clusters increases linearly with community size, until communities of about size 20, when the behavior appears to change (Fig. 5).”  
  Why this matters for SE_i: Supports multi-cluster structure and possible regime shifts in spatial coupling strength or neighborhood definitions [Data: Sources (3)].

- [E4] “T_{ij} = m_i p_{ij} ∝ (m_i m_j) / d_{ij}^D, (6) … Equation (6) is a standard gravity model with a power-law distance function. … the travel distance distribution is given by the same form: P(d) ∝ d^−D.”  
  Why this matters for SE_i: Gives an explicit flow template linking node “mass,” distance, and transfer magnitude [Data: Sources (809)].

- [E5] “P_new = 1 / (1 + λ(ln S + C)). (5)”  
  Why this matters for SE_i: Introduces a memory-dependent exploration probability that can modulate spatial interaction rates [Data: Sources (809)].

- [E6] “f_r ∝ (λ S)/r + 1 − λ. (4) … For λ = 1, we recover Zipf’s law … as the memory effect is intensified, a walker tends to travel among only a few locations.”  
  Why this matters for SE_i: Provides a rank-frequency form and qualitative effect of memory on spatial concentration of flows [Data: Sources (809)].

- [E7] “Network flow modelling. The traffic values were computed on all edges using equations (2–5) and compared with real traffic values…”  
  Why this matters for SE_i: Establishes that edge-based flow assignment operators are available and validated against data [Data: Sources (279)].

- [E8] “The PCCs … show a significant difference, 0.273 versus 0.639, indicating that travel time is a much better criterion for evaluating cost of travel than travel distance. … velocity classes 90-40-15 mph …”  
  Why this matters for SE_i: Supports choosing time-based costs and limited velocity classes in spatial operators [Data: Sources (279)].

- [E9] “A better agreement can be achieved if capacity limitation is taken into consideration … the highest obtained PCC is 0.752 when using travel time costs. … the iterations were stopped when condition (5) was satisfied.”  
  Why this matters for SE_i: Capacity constraints and iterative convergence criteria improve realism and stability [Data: Sources (279)].

- [E10] “The traffic values were generated using the weighted betweenness centrality type expression (2). … the shape of the traffic density … is lognormal. … p(b) ∼ (1/b) ∫ dk P(k) Ψ_r( log b − log β_r − log k ).”  
  Why this matters for SE_i: Motivates centrality-based operators and suggests expected distributional properties of flows [Data: Sources (279)].

- [E11] “overlap is 25.7% … The overall network overlap … is 8.55%. We … set Γ = 0.08 as a lower bound … the probability of having less than 10% overlap is quite high…”  
  Why this matters for SE_i: Quantifies realistic inter-layer overlap to calibrate dual-network coupling terms [Data: Sources (160)].

- [E12] “For simulation, we assume a population of 1000 nodes … β_r = 0.1, γ = 0.5, and β_t = 0 to 2.5. We assume Δt = 10^−6. This corresponds to γ^−1 = 2 days, β_r^−1 = 10 days, and β_t^−1 ranging over 10 to 0 days … The optimal network case … Γ = 1 … minimal network overlap, Γ = 0.08 … For every simulation … one initial randomly selected infectious case. … K = 10.”  
  Why this matters for SE_i: Provides concrete parameterization and timescale mapping for network-mediated spread coupling [Data: Sources (160)].

- [E13] “Figure 5 presents how the peak of the epidemic is affected by the overlap Γ … In general, the greater the overlap … the more effective contact tracing is. More precisely, the maximum number of infected people decays exponentially with the network overlap.”  
  Why this matters for SE_i: Supports overlap-dependent coupling strength in multi-layer interaction terms [Data: Sources (160)].

- [E14] “Why does contact tracing work with such low overlap? … explained by a simple fact: when using contact tracing, an increase in the number of infected people causes an increase in the tracing effort. This adaptation phenomenon is not present when only random tracing is used.”  
  Why this matters for SE_i: Introduces adaptive coupling driven by state-dependent resource allocation [Data: Sources (162)].

- [E15] “We measure the tracing effort defined as the sum … eff = eff_r + eff_t = ∑_{n ∈ (I ∪ S)} β_r + ∑_{n ∈ (Neighbour_t(T) ∩ (S ∪ I))} K_t(n)β_t (5)”  
  Why this matters for SE_i: Gives an explicit operator summing over neighbors in a tracing network, parameterized by β_r, β_t, and K_t(n) [Data: Sources (162)].

- [E16] “Considering … only random tracing, β_r = 0.20 and β_t = 0 … the number of infected nodes grows up to 300 … With the addition of contact tracing (β_r = 0.20 and β_t = 2) … significant reduction in the number of infected cases (below 45 …)”  
  Why this matters for SE_i: Empirical effect size supports including targeted neighbor-based coupling to damp peaks [Data: Sources (162)].

## 3) A “menu” of candidate building blocks for SE_i (no invention)

- Distance-decay kernel (power law)
  - What it does: Sets interaction strength to decay with distance, P(d) ∼ d^−1.5 [Data: Sources (3)].
  - Templates: [E2].
  - Constraints/assumptions: Empirical from social ties; suggests stronger local coupling than distant [Data: Sources (3)].

- Gravity model coupling
  - What it does: Generates flows between nodes proportional to node “mass” and inverse power of distance, T_{ij} ∝ (m_i m_j)/d_{ij}^D; implies P(d) ∝ d^−D [Data: Sources (809)].
  - Templates: [E4].
  - Constraints/assumptions: Fractal spatial domains; m_i, m_j represent population at locations; power-law distance effect [Data: Sources (809)].

- Memory-modulated mobility operator
  - What it does: Adjusts probability of exploring new vs. returning locations via P_new = 1 / (1 + λ(ln S + C)) and rank-frequency f_r ∝ (λ S)/r + 1 − λ [Data: Sources (809)].
  - Templates: [E5], [E6].
  - Constraints/assumptions: λ tunes memory strength; higher λ concentrates movement among fewer locations [Data: Sources (809)].

- Cost-based routing/flow assignment on networks
  - What it does: Computes edge flows using equations (2–5) with travel-time cost (higher PCC than distance) and capacity limits; uses limited velocity classes [Data: Sources (279)].
  - Templates: [E7], [E8], [E9].
  - Constraints/assumptions: Velocity classes (90–40–15 mph); iteration stopping on condition (5); capacity limitation improves correlation (PCC up to 0.752) [Data: Sources (279)].

- Centrality-driven flow (weighted betweenness)
  - What it does: Generates traffic values from a weighted betweenness expression; predicts lognormal traffic distributions linked to network structure [Data: Sources (279)].
  - Templates: [E10].
  - Constraints/assumptions: Distribution arises from convolution over P(k) and shell-size noise Ψ_r; relates to structural variability [Data: Sources (279)].

- Dual-network overlap coupling (epidemic/tracing)
  - What it does: Modulates effectiveness of neighbor-based interventions by overlap Γ between layers; peak infections decay exponentially with Γ [Data: Sources (160)].
  - Templates: [E12], [E13].
  - Constraints/assumptions: Realistic Γ can be low (~0.08); simulate with β_r, β_t, γ, Δt, K on a 1000-node population [Data: Sources (160)].

- Adaptive neighbor-based effort term
  - What it does: State-dependent tracing effort summing over nodes and their tracing-network neighbors: eff = ∑_{n∈(I∪S)} β_r + ∑_{n∈(Neighbour_t(T)∩(S∪I))} K_t(n)β_t [Data: Sources (162)].
  - Templates: [E15].
  - Constraints/assumptions: Explains effectiveness at low Γ via adaptation; parameterized by β_r, β_t, K_t(n) [Data: Sources (162)].

- Spatial dispersion and clustering priors
  - What it does: Uses geographic span D(s) and observed scaling of cluster counts with community size to inform heterogeneity in spatial coupling [Data: Sources (3)].
  - Templates: [E1], [E3].
  - Constraints/assumptions: Real spans are smaller than null-model spans; behavior change around community size ~20 [Data: Sources (3)].

## 4) Symbols & parameters index (only if supported)

- D(s): Geographic span of community s; average distance of members from community center; measured in distance units [Data: Sources (3)].
- n_s: Number of members in community s (denominator in D(s)) [Data: Sources (3)].
- (X_s, Y_s), (x_i, y_i): Community center coordinates and member coordinates used in D(s) [Data: Sources (3)].
- P(d): Connection probability as a function of distance; empirically decays as d^−1.5 [Data: Sources (3)].
- k (clusters): Number of clusters in k-means; optimal k chosen via AIC; number of spatial clusters increases linearly with size until ~20 [Data: Sources (3)].
- λ: Memory effect parameter; higher λ increases concentration of visits [Data: Sources (809)].
- S: Appears with λ in P_new and f_r; context: number of locations (as used in the model) [Data: Sources (809)].
- r: Rank in rank-frequency relation f_r [Data: Sources (809)].
- f_r: Rank-frequency of visits; f_r ∝ (λ S)/r + 1 − λ; Zipf’s law at λ = 1 [Data: Sources (809)].
- P_new: Probability to choose a new location; P_new = 1 / (1 + λ(ln S + C)) [Data: Sources (809)].
- C: Constant appearing in P_new [Data: Sources (809)].
- T_{ij}: Total number of traveling steps from i to j (population-level flow) [Data: Sources (809)].
- m_i, m_j: Number of individuals (mass) at locations i and j in gravity model [Data: Sources (809)].
- d_{ij}: Distance between locations i and j [Data: Sources (809)].
- D (fractal dimension): Exponent controlling distance effect in gravity model and travel-distance distribution [Data: Sources (809)].
- Γ: Network overlap parameter between layers; empirical overall overlap ~0.0855; minimal used Γ=0.08 [Data: Sources (160)].
- β_r, β_t: Tracing intensities for random tracing and contact tracing; values used include β_r = 0.1 (or 0.20 in examples), β_t ∈ [0, 2.5] [Data: Sources (160, 162)].
- γ: Recovery/removal rate; γ = 0.5 (γ^−1 = 2 days) in simulations [Data: Sources (160)].
- Δt: Simulation time step; Δt = 10^−6 (chosen < 1 second) [Data: Sources (160)].
- K: Parameter used in simulations (e.g., K = 10) [Data: Sources (160)].
- Neighbour_t(T): Set of neighbors in the tracing network of traced nodes T [Data: Sources (162)].
- K_t(n): Parameter multiplying β_t for node n in tracing effort (contact tracing intensity per neighbor) [Data: Sources (162)].
- eff, eff_r, eff_t: Total, random, and contact tracing efforts; eff = eff_r + eff_t with explicit sums over nodes and neighbors [Data: Sources (162)].
- ζ: Overall multiplying factor set to match mean traffic distribution in the model [Data: Sources (279)].
- PCC: Pearson correlation coefficient used to assess model–data agreement (e.g., 0.273 vs. 0.639 vs. 0.752) [Data: Sources (279)].
- Velocity classes: Typical travel velocities grouped as 90–40–15 mph [Data: Sources (279)].
- Range limits: Example limits in capacity-limited cases (e.g., 100 km, 100 min) [Data: Sources (279)].
- b: Betweenness variable in traffic density derivation; p(b) given by convolution over P(k) and Ψ_r [Data: Sources (279)].

