import streamlit as st
import numpy as np
import pandas as pd
import secrets
import time
import io
from scipy.stats import truncnorm

# ==========================================
# MOTORUL DE CALCUL - 20 STRATEGII DE ELITĂ
# ==========================================
class LotoMasterEngine:
    @staticmethod
    def create_pool(total, fixed_limit, extra_count):
        fixed_part = set(range(1, fixed_limit + 1))
        remaining = list(set(range(1, total + 1)) - fixed_part)
        extra_count = min(extra_count, len(remaining))
        extra_part = secrets.SystemRandom().sample(remaining, extra_count)
        return sorted(list(fixed_part.union(set(extra_part))))

    @staticmethod
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0: return False
        return True

    @staticmethod
    def generate_logic(p, draw, strategy, i_index):
        n = len(p)
        rng = np.random.default_rng()
        
        # Selectie Strategie
        if strategy == "Criptografic (High Security)":
            variant = secrets.SystemRandom().sample(list(p), draw)
        elif strategy == "Gaussian Distribution":
            mu, sigma = n/2, n/4
            idx = truncnorm((0-mu)/sigma, (n-1-mu)/sigma, loc=mu, scale=sigma).rvs(draw)
            variant = p[idx.astype(int)]
        elif strategy == "Quantum Step":
            step = max(1, n // (draw + 1))
            variant = [p[(secrets.randbelow(step) + j*step) % n] for j in range(draw)]
        elif strategy == "Prime Affinity":
            w = np.array([1.6 if LotoMasterEngine.is_prime(x) else 1.0 for x in p])
            variant = rng.choice(p, size=draw, replace=False, p=w/w.sum())
        elif strategy == "Fibonacci Sequence":
            fib = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
            w = np.array([1.5 if (x in fib) else 1.0 for x in p])
            variant = rng.choice(p, size=draw, replace=False, p=w/w.sum())
        elif strategy == "Inverse Density":
            w = np.abs(np.arange(n) - n/2) + 0.1
            variant = rng.choice(p, size=draw, replace=False, p=w/w.sum())
        elif strategy == "Entropy Chaos":
            px = p.copy(); [rng.shuffle(px) for _ in range(5)]
            variant = px[:draw]
        elif strategy == "Monte Carlo Stability":
            cands = [sorted(rng.choice(p, size=draw, replace=False)) for _ in range(10)]
            variant = cands[secrets.randbelow(10)]
        elif strategy == "Poisson Random":
            idx = np.random.poisson(n/2, draw*3)
            idx = np.unique(idx[idx < n])[:draw]
            variant = p[idx.astype(int)]
        elif strategy == "Weighted Balance (L-H)":
            w = np.linspace(1, 2, n) if i_index % 2 == 0 else np.linspace(2, 1, n)
            variant = rng.choice(p, size=draw, replace=False, p=w/w.sum())
        elif strategy == "Arithmetic Progression":
            s = secrets.randbelow(max(1, n//2))
            variant = p[np.arange(s, n, max(1, n//draw))[:draw]]
        elif strategy == "Markov Chain Lite":
            idx = [secrets.randbelow(n)]
            for _ in range(draw-1): idx.append((idx[-1] + secrets.randbelow(n//3)) % n)
            variant = p[idx]
        elif strategy == "Geometric Spacing":
            variant = p[np.linspace(0, n-1, draw, dtype=int)]
        elif strategy == "Harmonic Mean Filter":
            variant = rng.choice(p, size=draw, replace=False)
        elif strategy == "Predictive Trend":
            w = np.sin(np.linspace(0, np.pi, n)) + 1
            variant = rng.choice(p, size=draw, replace=False, p=w/w.sum())
        elif strategy == "Delta Gap Control":
            variant = sorted(rng.choice(p, size=draw, replace=False))
        elif strategy == "Bimodal Distribution":
            w = np.exp(-(np.linspace(-2,2,n)**2)) + np.exp(-(np.linspace(-1,1,n)**2))
            variant = rng.choice(p, size=draw, replace=False, p=w/w.sum())
        elif strategy == "Logarithmic Scale":
            w = np.log1p(np.arange(n)) + 0.1
            variant = rng.choice(p, size=draw, replace=False, p=w/w.sum())
        elif strategy == "Mirror Reflection":
            idx = [i_index % n, (n - 1 - i_index) % n]
            while len(idx) < draw: idx.append(secrets.randbelow(n))
            variant = p[idx[:draw]]
        else: # "Stochastic Oscillator"
            w = np.random.uniform(0.5, 1.5, n)
            variant = rng.choice(p, size=draw, replace=False, p=w/w.sum())

        fv = sorted(list(set(variant)))
        while len(fv) < draw:
            ex = secrets.SystemRandom().choice(p)
            if ex not in fv: fv.append(ex)
        return sorted(fv)

# ==========================================
# INTERFAȚĂ UTILIZATOR
# ==========================================
def main():
    st.set_page_config(page_title="Ultra Loto 20 Strat", layout="wide")
    
    # CSS pentru chenar cu scroll (Pasul 6 & 9)
    st.markdown("""
        <style>
        .stCodeBlockContainer { max-height: 350px !important; overflow-y: auto !important; border: 1px solid #00cc66; }
        </style>
    """, unsafe_allow_html=True)

    st.title("🛡️ Generator Loto Profesional - 20 Strategii & Mix Custom")

    strats_list = [
        "Criptografic (High Security)", "Gaussian Distribution", "Quantum Step", "Prime Affinity",
        "Fibonacci Sequence", "Inverse Density", "Entropy Chaos", "Monte Carlo Stability",
        "Poisson Random", "Weighted Balance (L-H)", "Arithmetic Progression", "Markov Chain Lite",
        "Geometric Spacing", "Harmonic Mean Filter", "Predictive Trend", "Delta Gap Control",
        "Bimodal Distribution", "Logarithmic Scale", "Mirror Reflection", "Stochastic Oscillator"
    ]

    with st.sidebar:
        st.header("⚙️ Configurare Urnă")
        total = st.number_input("Bile în urnă", 1, 1000, 80)
        draw = st.number_input("Numere extrase", 1, total, 12)
        st.divider()
        st.subheader("🧬 Logică Pool (Pasul 8)")
        f_lim = st.number_input("Interval Fix (1-X):", 1, total, 25)
        e_cnt = st.number_input("Extra Random:", 0, total-f_lim, 15)
        st.divider()
        v_count = st.number_input("Variante de generat", 1, 100000, 15000)
        
        st.subheader("🎯 Mixează Strategiile")
        sel_strats = [s for s in strats_list if st.checkbox(s, value=(s == strats_list[0]))]

    if not sel_strats:
        st.warning("⚠️ Bifează cel puțin o strategie!")
        return

    tab_gen, tab_man = st.tabs(["🚀 Generator Principal", "📥 Mod Manual"])

    with tab_gen:
        if st.button("LANCEAZĂ GENERAREA", use_container_width=True):
            t1 = time.time()
            pool = LotoMasterEngine.create_pool(total, f_lim, e_cnt)
            pool_arr = np.array(pool)
            
            variants = []
            per_s = v_count // len(sel_strats)
            
            for s_name in sel_strats:
                for i in range(per_s):
                    v = LotoMasterEngine.generate_logic(pool_arr, draw, s_name, i)
                    variants.append(" ".join(map(str, v)))
            
            while len(variants) < v_count:
                v = LotoMasterEngine.generate_logic(pool_arr, draw, sel_strats[0], 0)
                variants.append(" ".join(map(str, v)))
            
            secrets.SystemRandom().shuffle(variants)
            
            # Creare text final
            out_io = io.StringIO()
            for i, v in enumerate(variants):
                out_io.write(f"{i+1}, {v}\n")
            
            full_txt = out_io.getvalue()
            st.success(f"Gata! {v_count} variante generate în {time.time()-t1:.3f}s")

            st.subheader("📋 Chenar Rezultate (Scroll & Copy)")
            st.info("Apar primele ~10 variante, dar butonul 'Copy' ia toate cele 15.000+.")
            st.code(full_txt, language='text') 

            st.download_button("📥 DESCARCĂ .TXT COMPLET", full_txt, "loto_export.txt", "text/plain", use_container_width=True)

    with tab_man:
        m_in = st.text_area("Input manual:", height=200)
        if st.button("Procesează"):
            res = [f"{i+1}, {l.strip()}" for i, l in enumerate(m_in.split('\n')) if l.strip()]
            if res: st.code("\n".join(res), language='text')

if __name__ == "__main__":
    main()
