import streamlit as st
import numpy as np
import pandas as pd
import secrets
import time
import io
from scipy.stats import truncnorm

# ==========================================
# MOTORUL DE CALCUL - HIBRID MULTI-STRATEGIE
# ==========================================
class LotoHybridEngine:
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
    def generate_single_variant(p, draw, strategy):
        n_pool = len(p)
        rng = np.random.default_rng()
        
        # Logica interna pentru fiecare strategie (extrase din pool-ul P)
        if strategy == "Criptografic":
            variant = secrets.SystemRandom().sample(list(p), draw)
        elif strategy == "Gaussian":
            mu, sigma = n_pool/2, n_pool/4
            idx = truncnorm((0-mu)/sigma, (n_pool-1-mu)/sigma, loc=mu, scale=sigma).rvs(draw)
            variant = p[idx.astype(int)]
        elif strategy == "Quantum Leap":
            step = max(1, n_pool // (draw + 1))
            variant = [p[(secrets.randbelow(step) + j*step) % n_pool] for j in range(draw)]
        elif strategy == "Prime Affinity":
            w = np.array([1.5 if LotoHybridEngine.is_prime(x) else 1.0 for x in p])
            variant = rng.choice(p, size=draw, replace=False, p=w/w.sum())
        elif strategy == "Fibonacci":
            fib = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
            w = np.array([1.8 if (x in fib) else 1.0 for x in p])
            variant = rng.choice(p, size=draw, replace=False, p=w/w.sum())
        elif strategy == "Inverse Density":
            w = np.abs(np.arange(n_pool) - n_pool/2) + 0.1
            variant = rng.choice(p, size=draw, replace=False, p=w/w.sum())
        elif strategy == "Entropy Shuffle":
            px = p.copy()
            for _ in range(5): rng.shuffle(px)
            variant = px[:draw]
        elif strategy == "Monte Carlo":
            candidates = [sorted(rng.choice(p, size=draw, replace=False)) for _ in range(5)]
            variant = candidates[secrets.randbelow(5)]
        elif strategy == "Delta Distance":
            variant = sorted(rng.choice(p, size=draw, replace=False)) # Spacing automat prin sortare
        elif strategy == "Poisson":
            lam = n_pool / 2
            idx = np.random.poisson(lam, draw*3)
            idx = np.unique(idx[idx < n_pool])[:draw]
            variant = p[idx.astype(int)]
        elif strategy == "Weighted Balance":
            w = np.linspace(1, 2, n_pool)
            variant = rng.choice(p, size=draw, replace=False, p=w/w.sum())
        elif strategy == "Harmonic":
            variant = rng.choice(p, size=draw, replace=False)
        elif strategy == "Geometric":
            indices = np.linspace(0, n_pool - 1, draw, dtype=int)
            variant = p[indices]
        elif strategy == "Arithmetic":
            start = secrets.randbelow(max(1, n_pool // 2))
            variant = p[np.arange(start, n_pool, max(1, n_pool//draw))[:draw]]
        else: # Markov Lite
            idx = [secrets.randbelow(n_pool)]
            for _ in range(draw-1):
                idx.append((idx[-1] + secrets.randbelow(max(1, n_pool//3))) % n_pool)
            variant = p[idx]

        # Validare unicitate si marime
        final_v = sorted(list(set(variant)))
        while len(final_v) < draw:
            extra = secrets.SystemRandom().choice(p)
            if extra not in final_v: final_v.append(extra)
        return sorted(final_v)

# ==========================================
# INTERFATA UTILIZATOR
# ==========================================
def main():
    st.set_page_config(page_title="Ultra Loto Hybrid", layout="wide")
    st.title("🛡️ Generator Loto Profesional - Mix de Strategii")

    # Toate cele 15 strategii
    all_strategies = [
        "Criptografic", "Gaussian", "Quantum Leap", "Prime Affinity", 
        "Fibonacci", "Inverse Density", "Entropy Shuffle", "Monte Carlo",
        "Delta Distance", "Poisson", "Weighted Balance", "Harmonic", 
        "Geometric", "Arithmetic", "Markov Lite"
    ]

    with st.sidebar:
        st.header("⚙️ Configurare Sistem")
        total = st.number_input("Bile în urnă", 1, 1000, 80)
        draw = st.number_input("Numere extrase", 1, total, 12)
        
        st.divider()
        st.subheader("🧬 Parametri Pool (Pasul 8)")
        f_lim = st.number_input("Interval Fix (1-X):", 1, total, 25)
        e_cnt = st.number_input("Extra Random:", 0, total-f_lim, 15)
        
        st.divider()
        v_count = st.number_input("Variante de generat", 1, 100000, 15000)
        
        st.subheader("🎯 Alege Strategiile (MIX)")
        selected_strats = []
        for s in all_strategies:
            if st.checkbox(s, value=(s == "Criptografic")):
                selected_strats.append(s)

    if not selected_strats:
        st.warning("⚠️ Te rugăm să bifezi cel puțin o strategie!")
        return

    tab_gen, tab_man = st.tabs(["🚀 Generator Multi-Strategie", "📥 Manual"])

    with tab_gen:
        if st.button("LANCEAZĂ GENERAREA HIBRIDĂ", use_container_width=True):
            t1 = time.time()
            pool = LotoHybridEngine.create_pool(total, f_lim, e_cnt)
            pool_arr = np.array(pool)
            
            final_results = []
            # Distribuim numarul de variante per strategie
            per_strat = v_count // len(selected_strats)
            
            for s_name in selected_strats:
                for _ in range(per_strat):
                    v = LotoHybridEngine.generate_single_variant(pool_arr, draw, s_name)
                    final_results.append(" ".join(map(str, v)))
            
            # Umplem restul pana la v_count daca exista rest la impartire
            while len(final_results) < v_count:
                v = LotoHybridEngine.generate_single_variant(pool_arr, draw, selected_strats[0])
                final_results.append(" ".join(map(str, v)))
            
            # Shuffle final pentru a mixa strategiile intre ele
            secrets.SystemRandom().shuffle(final_results)
            
            # Formatare Finala DataFrame
            df = pd.DataFrame({
                "ID": [f"{i+1}," for i in range(len(final_results))],
                "COMBINATIE": final_results
            })
            
            t2 = time.time()
            st.success(f"Gata! {v_count} variante mixate generate în {t2-t1:.3f} secunde.")
            st.dataframe(df, height=500, use_container_width=True, hide_index=True)

            # Export TXT
            txt = io.StringIO()
            for r in df.itertuples():
                txt.write(f"{r.ID} {r.COMBINATIE}\n")
            
            st.download_button("📥 DESCARCĂ .TXT MIXAT", txt.getvalue(), "loto_mix_export.txt", "text/plain", use_container_width=True)

    with tab_man:
        manual_in = st.text_area("Input manual (ID, numere):", height=300)
        if st.button("Procesează Manual"):
            res = [{"ID": f"{i+1},", "COMBINATIE": l.strip()} for i, l in enumerate(manual_in.split('\n')) if l.strip()]
            if res:
                st.dataframe(pd.DataFrame(res), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
