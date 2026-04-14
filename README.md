# Matali : Hidden Markov Model Hospital Routing Simulator

**AFMC Illuminati Hackathon 2026 | Problem Statement B**  
*Saving Seconds, Saving Lives: Affordable Innovations in Emergency Care*

**Live Demo**  
[Open Interactive Simulator](https://anirudhgangadharan.github.io/matali/)  
(Toggle injury, severity, dispatch load → watch real-time HMM routing + graphs)

**What it does**  
Voice-first AI toll-free system that routes emergency calls to the *optimal* hospital using HMM.  
Predicts hospital state (Available / Saturated / Diverting) at ETA using dynamic transition matrix + forward projection.  
Self-balances load across hospitals. No API needed.

**Core Math**  
Uᵢ = P(Available at arrival) × Cᵢ − λ × ETAᵢ  
(with corrected exponential decay a₁₂(δ) and matrix exponentiation)

**Files**
- `Matali.ipynb` — full Colab simulation (20 calls, 7 figures)
- `matali.py` — clean Python version
- `matali-hidden-markov-model-simulator.html` — standalone interactive demo (Chart.js)

**Key Results** (from simulation)
- PHC saturates after ~3 dispatches; Medical College absorbs 10+
- System naturally load-balances (Fig 5)
- Severity changes optimal hospital (Fig 7)

**Team**  
Anirudh Gangadharan | GIMS Gadag | 3rd MBBS

**Built for** AFMC Illuminati Hackathon 2026
