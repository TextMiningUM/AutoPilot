# AutoPilot / COLREG — referentielijst

Samengesteld op 2026-09-25 uit de chat. Per entry: bronvermelding, link (alleen als in de sessie gevonden), BibTeX-sleutel uit `autopilot_colreg_references.bib`, en waarom de bron voor het project relevant is. Entries met **VERIFY** hebben velden (auteurs/volume/jaar) die nog gecontroleerd moeten worden voordat je ze citeert.

---

## 1. Imazu-benchmark en DRL-papers die hem gebruiken

| # | Referentie | Link | BibTeX-sleutel | Relevantie voor het project |
|---|---|---|---|---|
| 1 | Sawada, Sato & Majima (2021). *Automatic ship collision avoidance using deep reinforcement learning with LSTM in continuous action spaces.* J. Marine Science and Technology 26:509–524. Open access. | [Springer](https://link.springer.com/article/10.1007/s00773-020-00755-0) · [PDF](https://scispace.com/pdf/automatic-ship-collision-avoidance-using-deep-reinforcement-4mpps56bvi.pdf) | `sawada2021automatic` | **Hoofdreferentie.** Table 4 = jouw Imazu-missies (eigen schip (−6,0) NM, 12 kt, targets TCPA 30 min). Nomoto-model K=0,05, T=50 s, T_E=2,5 s; roer ±10°; beslissing elke 10 s; veilige afstand 0,5 NM + bow-crossing 1,0 NM; targets varen recht. |
| 2 | Imazu, H. (1987). *Research on Collision Avoidance Manoeuvre.* PhD thesis, Univ. of Tokyo (Japans). | geen link gevonden | `imazu1987research` | Oorspronkelijke 42 situaties (22 + 20); de tweede set van 20 is nauwelijks in Engelstalige literatuur gereproduceerd → kandidaat "Imazu-2"-benchmark. |
| 3 | Cai & Hasegawa (2013). *Evaluating of marine traffic simulation system through Imazu problem.* Proc. JASNAOE 17:191–194. | geen link gevonden | `cai2013evaluating` | Bron van de moderne Imazu-opzet die Sawada overneemt; merkt op dat het probleem makkelijker wordt als targets zelf uitwijken. |
| 4 | Xie et al. (2023). *Generalized Behavior Decision-Making Model for Ship Collision Avoidance via RL.* JMSE 11(2):273. **VERIFY** auteurs. | [MDPI](https://www.mdpi.com/2077-1312/11/2/273) | `xie2023generalized` | 3-DOF Nomoto-model, discrete roerhoeken, grid sensor + OZT; Imazu als leerscenario. |
| 5 | *Method for collision avoidance based on DRL with path-speed control for an autonomous ship* (2023). Int. J. Naval Arch. Ocean Eng. **VERIFY** auteurs. | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2092678223000687) | `pathspeed2023drl` | Koers-only: 6/20 Imazu-cases falen; koers+snelheid: 20/20 → argument om snelheidsacties te behouden als variant. Vervolg op Chun et al. (2021). |
| 6 | *Autonomous collision avoidance system in a multi-ship environment based on PPO* (2023). Ocean Engineering. **VERIFY** auteurs. | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0029801823001634) | `ppo2023multiship` | PPO op Imazu; schakelt tussen path-tracking en avoidance-modus. |
| 7 | *A Multi-Ship Collision Avoidance Algorithm Using Data-Driven Multi-Agent DRL* (2023). JMSE 11(11):2101. **VERIFY** auteurs. | [DOI](https://doi.org/10.3390/jmse11112101) | `madrl2023multiship` | Bevat een "extended encounter scenario library based on Imazu problem" (Fig. 11); AIS-gedreven. |
| 8 | Pan, Zhang, Wang & Kang (2025). *DRL model for multi-ship collision avoidance decision making.* Scientific Reports 15:21250. | [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12219284/) | `pan2025deep` | Recente DRL-referentie op Imazu-achtige scenario's. |
| 9 | *Reinforcement Learning-Based Autonomous Collision Avoidance for Ships in Realistic Physical Environments* (Springer-hoofdstuk, 2025/26). **VERIFY** auteurs/boek. | [Springer](https://link.springer.com/chapter/10.1007/978-981-95-8408-6_4) | `rl2025realistic` | Vergelijkt VO, DQN en LSTM-DQN in één omgeving; hexagonaal ship domain. |
| 10 | *COLREGs Compliant Collision Avoidance and Grounding Prevention for Autonomous Marine Navigation* (arXiv 2603.02484, 2026). **VERIFY** auteurs. | [arXiv](https://arxiv.org/pdf/2603.02484) | `arxiv2603_02484` | VO-gebaseerd; Imazu-cases met ondiep water; head-on-drempel ±5°. |
| 11 | *Sim2Sea: Sim-to-Real Policy Transfer for Maritime Vessel Navigation in Congested Waters* (arXiv 2603.04057, 2026). **VERIFY** auteurs. | [arXiv](https://arxiv.org/pdf/2603.04057) | `sim2sea2026` | Gebruikt Paulig's open-source MMG-implementatie. |
| 12 | Shen et al. (2019). *Automatic collision avoidance of multiple ships based on deep Q-learning.* Applied Ocean Research 86:268–288. | geen link gevonden | `shen2019automatic` | DQN-baseline; vollere hydrodynamische modellen. |
| 13 | Zhao & Roh (2019). *COLREGs-compliant multiship collision avoidance based on DRL.* Ocean Engineering. **VERIFY** volume. | geen link gevonden | `zhao2019colregs` | Veelgeciteerde DRL-baseline met kinematisch model. |
| 14 | Nomoto (1960). *Analysis of Kempf's standard maneuver test and proposed steering quality indices.* Proc. 1st Symp. Ship Manoeuvrability, 275–304. | — | `nomoto1960analysis` | Het bewegingsmodel dat je in de simulator overneemt. |

---

## 2. Deterministische baselines (rule-based, VO, MPC) en code

| # | Referentie | Link | BibTeX-sleutel | Relevantie voor het project |
|---|---|---|---|---|
| 15 | Benjamin & Woerner. *MOOS-IvP COLREGS behaviors* (`BHV_AvdColregsV17/V19/V22`, `lib_behaviors-colregs`), GPL. | [SVN-bron](https://oceanai.mit.edu/svn/moos-ivp-aro/releases/moos-ivp-22.8/ivp/src/lib_behaviors-colregs) · [Advanced COLREGS project](https://oceanai.mit.edu/pavlab/pdfs/proj_advcolregs.pdf) | `moosivp_colregs` | **Aanbevolen hoofdbaseline.** Gepubliceerd, veldgetest, draait al in je MOOS-IvP-omgeving. Projectbeschrijving noemt zelf de zwakke plekken (multi-contact, uncooperative contacts). |
| 16 | Woerner, Benjamin, Novitzky & Leonard (2018/2019). *Quantifying protocol evaluation for autonomous collision avoidance.* Autonomous Robots. **VERIFY** volume/pagina's. | [catalogusrecord](https://openaccess.library.uitm.edu.my/Record/mit_116295/Details) | `woerner2018quantifying` | COLREG-compliance-metrieken; basis voor je auditor-codes. |
| 17 | Woerner et al. (2016). *Collision avoidance road test for COLREGS-constrained autonomous vehicles.* OCEANS 2016. | geen link gevonden | `woerner2016collision` | Zes-scenario testset die Sawada naast Imazu noemt. |
| 18 | Kuwata et al. (2014). *Safe maritime autonomous navigation with COLREGS, using velocity obstacles.* IEEE J. Oceanic Eng. 39(1):110–119. | geen link gevonden | `kuwata2014safe` | VO-baseline; geen publieke code, formules staan in Sawada §2.1. |
| 19 | Johansen, Perez & Cristofaro (2016). *Ship collision avoidance and COLREGS compliance using simulation-based control behavior selection…* IEEE T-ITS 17(12):3407–3422. | geen link gevonden | `johansen2016ship` | SB-MPC: sterkste deterministische aanpak; zelf herimplementeren op je simulator. |
| 20 | Meyer, Heiberg, Rasheed & San (2020). *COLREG-Compliant Collision Avoidance for USV using DRL.* IEEE Access 8:165344–165364. | [arXiv](https://arxiv.org/pdf/2006.09540) | `meyer2020colreg` | Goede literatuurlijst van DWA/VO/MPC-baselines (NTNU). |
| 21 | Eriksen, Bitar, Breivik & Lekkas (2019/2020). *Hybrid Collision Avoidance for ASVs Compliant with COLREGs Rules 8 and 13–17.* **VERIFY** journal/jaar. | [arXiv](https://arxiv.org/pdf/1907.00198) | `eriksen2019hybrid` | Drielaags COLAV; branching-course MPC-referenties. |
| 22 | *VORRT-COLREGs* (arXiv 2109.00862, 2021). **VERIFY** auteurs. | [arXiv](https://arxiv.org/pdf/2109.00862) | `vorrt2021` | VO + RRT planner; overzicht van klassieke methoden. |
| 23 | *MPC-based path planning for ship collision avoidance under COLREGS* (2022). **VERIFY** auteurs/journal. | [ResearchGate](https://www.researchgate.net/publication/365588516_MPC-based_path_planning_for_ship_collision_avoidance_under_COLREGS) | `mpc2022pathplanning` | Merkt op dat targets in de meeste methoden koers houden ("niet realistisch"). |
| 24 | *Collision avoidance for ASVs through trajectory planning: MPC with COLREGs-compliant nonlinear constraints* (2022). **VERIFY** auteurs. | [ResearchGate](https://www.researchgate.net/publication/362491358_Collision_avoidance_for_ASVs_through_trajectory_planning_MPC_with_COLREGs-compliant_nonlinear_constraints) | `mpc2022nonlinear` | Timing/duur/omvang van manoeuvres expliciet gestuurd (Rule 8 "ample time"). |
| 25 | *Multi-Ship Control and Collision Avoidance Using MPC and RBF-Based Trajectory Predictions* (2021). **VERIFY** auteurs/journal. | [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8588155/) | `mpc2021rbf` | AIS-getrainde trajectvoorspelling; Miami-case. |
| 26 | Code: **atanu-dip/maritime-cas-verification** (2026). | [GitHub](https://github.com/atanu-dip/maritime-cas-verification) | `atanudip2026mcas` | Nomoto, 5 strategieën incl. rule-based COLREGS en MPC, Scenic + CMA-ES/BO-falsificatie. Kernbevinding: rule-based 0,68 compliant maar 57% veilig als stand-on. Klein en niet onderhouden → lezen, niet op leunen. |
| 27 | Code: **MengGuo/RVO_Py_MAS** — VO/RVO/HRVO in Python. | [GitHub](https://github.com/MengGuo/RVO_Py_MAS) | `guo_rvo_py_mas` | Startpunt voor een VO-baseline op je kinematica. |
| 28 | Code: **ppotoc/MPC-Autonomous-Ship-Navigation** (MATLAB). **VERIFY** auteursnaam. | [GitHub](https://github.com/ppotoc/MPC-Autonomous-Ship-Navigation) | `ppotoc_mpc` | MPC met COLREG-regels en kaartobstakels. |
| 29 | Code: **nikpau/mmgdynamics** — Paulig (2024), MMG-standaardmodel. | [GitHub](https://github.com/nikpau/mmgdynamics) | `paulig2024mmg` | Als je later van Nomoto naar MMG wilt. |

---

## 3. Scenario-generatie, "onmogelijke" missies en falsificatie

| # | Referentie | Link | BibTeX-sleutel | Relevantie voor het project |
|---|---|---|---|---|
| 30 | *Randomly Testing an Autonomous Collision Avoidance System with Real-World Ship Encounter Scenario from AIS Data* (2022). JMSE 10(11):1588. **VERIFY** auteurs. | [DOI](https://doi.org/10.3390/jmse10111588) | `zhang2022randomly` | **Kernbron voor "onmogelijke" missies:** systeem dat alle 22 Imazu-cases haalt faalt op AIS-cases waar range/DCPA te klein zijn om te handelen; noemt VHF-afstemming als menselijke oplossing. Vermeldt ook Imazu's 42 situaties en de sets van Johansen (15), Chen (15), Paul. |
| 31 | Bolbot et al. (2022). *Automatic traffic scenarios generation for autonomous ships collision avoidance system testing.* Ocean Engineering. **VERIFY** auteurs. | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0029801822007016) | `bolbot2022automatic` | Sobol-sampling + risicovector + clustering; systematische dekking van de scenario-ruimte. |
| 32 | Wang, Huang, Liu, Zhou, Yuan, Xin & Wu (2024). *Ship encounter scenario generation for collision avoidance algorithm testing based on AIS data.* Ocean Engineering 291:116436. | [ResearchGate](https://www.researchgate.net/publication/377047307_Ship_encounter_scenario_generation_for_collision_avoidance_algorithm_testing_based_on_AIS_data) | `wang2024ship` | AIS → testscenario-bibliotheek "op basis van het Imazu-idee". |
| 33 | *Generation and complexity analysis of ship encounter scenarios using AIS data for collision avoidance algorithm testing* (2024). Ocean Engineering. **VERIFY** auteurs. | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0029801824023722) | `complexity2024encounter` | Complexiteitsmaten voor encounters; verwijst naar Zhu et al. |
| 34 | Zhu, F. et al. (2024). *A high-risk test scenario adaptive generation algorithm for ship autonomous collision avoidance decision-making based on RL.* Ocean Engineering. **VERIFY** auteurs. | geen link gevonden | `zhu2024highrisk` | RL-agent die faal-scenario's voor een CA-methode zoekt — directe methodologische basis voor je "faal-missies". |
| 35 | Simula (2024). *Towards Automated Test Scenario Generation for Assuring COLREGs Compliance of ASVs.* MODELS 2024 Companion. **VERIFY** auteurs. | [ACM](https://dl.acm.org/doi/10.1145/3640310.3674098) | `simula2024towards` | Functional → logical → concrete scenario-hiërarchie, overgenomen uit AV-testen. |
| 36 | Simula (2025). *Assessing Scene Generation Techniques for Testing COLREGS-Compliance of ASVs.* Proc. ACM on Software Engineering. **VERIFY** auteurs. | [ACM](https://dl.acm.org/doi/10.1145/3728919) | `simula2025assessing` | Search-based scene-generatie; vergelijkt generatietechnieken. |
| 37 | *From Vessel Trajectories to Safety-Critical Encounter Scenarios: A Generative AI Framework for Autonomous Ship Digital Testing* (arXiv 2603.28067, 2026). **VERIFY** auteurs. | [arXiv](https://arxiv.org/html/2603.28067v1) | `arxiv2603_28067` | Generatieve scenario's uit trajecten; goede literatuursectie over knowledge-driven vs data-driven scenario's. |

---

## 4. Hoe deze bronnen in het project passen

- **Simulator (Nomoto, 0,5 NM, bow-crossing):** #1, #14, #4.
- **Imazu-missies en vergelijkbaarheid:** #1 (Table 4), #2, #3, #5 (snelheidsvariant).
- **Baseline-ladder** (eigen oracle → MOOS AvdColregs → VO/OZT → SB-MPC): #15, #16, #18, #19, #27.
- **Faal-missies en falsificatie:** #30 (AIS-bewijs dat zulke geometrieën bestaan), #34 (RL-zoektocht), #26/#35/#36 (CMA-ES / search-based), #31–#33 (scenario-bibliotheken).
- **Argument voor de LLM-agent:** #30 noemt expliciet VHF-afstemming en abrupt draaiende targets als de gevallen waar geometrische systemen tekortschieten.
