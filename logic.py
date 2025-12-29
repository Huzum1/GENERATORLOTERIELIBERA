import streamlit as st
import numpy as np
import pandas as pd
import secrets
import time
import io
from scipy.stats import truncnorm

# ==========================================
# MOTORUL DE CALCUL - 20 STRATEGII & POOL CUSTOM
# ==========================================
class LotoMasterEngine:
    @staticmethod
    def create_custom_pool(total, range_min, range_max, extra_count):
        """Logica Pasul 8 Evoluat: Interval selectabil + Rest Random"""
        # 1. Cream setul din intervalul selectat de utilizator (ex: 25-45)
        selected_interval = set(range(range_min, range_max + 1))
        
        # 2. Identificam toate numerele ramase in urna (in afara intervalului)
        all_possible = set(range(1, total + 1))
        remaining_pool = list(all_possible - selected_interval)
        
        # 3. Extragem bilele extra din restul urnei
        extra_count = min(extra_count, len(remaining_pool))
        extra_part = secrets.SystemRandom().sample(remaining_pool, extra_count)
        
        # 4. Combinam si sortam pool-ul final
        return sorted(list(selected_interval.union(set(extra_part))))

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
        if n < draw: return sorted(list(p)) # Safety check

        # Selectie Strategie (Exemple din cele 20)
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
        elif strategy == "Entropy Chaos":
            px = p.copy(); [rng.shuffle(px) for _ in range(3)]
            variant = px[:draw]
        elif strategy == "Weighted Balance (L-H)":
            w = np.linspace(1, 2, n) if i_index % 2 == 0 else np.linspace(2, 1, n)
            variant = rng.choice(p, size=draw, replace=False, p=w/w.sum())
        elif strategy == "Delta Gap Control":
            variant = sorted(rng.choice(p, size=draw, replace=False))
        elif strategy == "Predictive Trend":
            w = np.sin(np.linspace(0, np.pi, n)) + 1
            variant = rng.choice(p, size=draw, replace=False, p=w/w.sum())
        else: # Standard / Fibonacci / Restul pana la 20
            variant = rng.choice(p, size=draw, replace=False)

        fv = sorted(list(set(variant)))
        while len(fv) < draw:
            ex = secrets.SystemRandom().choice(p)
            if ex not in fv: fv.append(ex)
        return sorted(fv)

# ==========================================
# INTERFAȚĂ UTILIZATOR
# ==========================================
def main():
    st.set_page_config(page_title="Ultra Loto Custom Range", layout="wide")
    
    st.markdown("""
        <style>
        .stCodeBlockContainer { max-height: 350px !important; overflow-y: auto !important; border: 2px solid #00cc66; border-radius: 8px; }
        .stButton>button { background-color: #00cc66; color: white; border-radius: 20px; font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)

    st.title("🎰 Generator Loto Master - Custom Pool Logic")

    # Toate strategiile (reintegrate)
    strats_list = [
        "Criptografic (High Security)", "Gaussian Distribution", "Quantum Step", "Prime Affinity",
        "Fibonacci Sequence", "Inverse Density", "Entropy Chaos", "Monte Carlo Stability",
        "Poisson Random", "Weighted Balance (L-H)", "Arithmetic Progression", "Markov Chain Lite",
        "Geometric Spacing", "Harmonic Mean Filter", "Predictive Trend", "Delta Gap Control",
        "Bimodal Distribution", "Logarithmic Scale", "Mirror Reflection", "Stochastic Oscillator"
    ]

    with st.sidebar:
        st.header("⚙️ Setări Urnă")
        total = st.number_input("Bile totale în joc", 1, 1000, 80)
        draw = st.number_input("Bile extrase / rundă", 1, total, 12)
        
        st.divider()
        st.subheader("🧬 Configurare Interval Pool")
        # NOU: Selectie interval custom
        c_min, c_max = st.slider("Selectează Intervalul Fix:", 1, total, (25, 45))
        e_cnt = st.number_input("Adaugă bile RANDOM din restul urnei:", 0, total-(c_max-c_min+1), 15)
        
        st.divider()
        v_count = st.number_input("Variante de generat", 1, 100000, 15000)
        
        st.subheader("🎯 Mix Strategii")
        sel_strats = [s for s in strats_list if st.checkbox(s, value=(s == strats_list[0]))]

    if not sel_strats:
        st.warning("⚠️ Bifează cel puțin o strategie!")
        return

    tab_gen, tab_man = st.tabs(["🚀 Generator Principal", "📥 Mod Manual"])

    with tab_gen:
        if st.button("LANCEAZĂ GENERAREA", use_container_width=True):
            t1 = time.time()
            # Pasul 8 actualizat
            final_pool = LotoMasterEngine.create_custom_pool(total, c_min, c_max, e_cnt)
            pool_arr = np.array(final_pool)
            
            st.write(f"✅ **Pool creat:** {len(final_pool)} numere unice (Interval {c_min}-{c_max} + {e_cnt} extra)")
            
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
            
            # Pregătire text final
            out_io = io.StringIO()
            for i, v in enumerate(variants):
                out_io.write(f"{i+1}, {v}\n")
            
            full_txt = out_io.getvalue()
            st.success(f"Gata! {v_count} variante generate în {time.time()-t1:.3f}s")

            st.subheader("📋 Chenar Rezultate (Scroll & Copy All)")
            st.code(full_txt, language='text') 

            st.download_button("📥 DESCARCĂ .TXT COMPLET", full_txt, "loto_export.txt", "text/plain", use_container_width=True)

    with tab_man:
        m_in = st.text_area("Input manual (ID, numere):", height=200)
        if st.button("Procesează"):
            res = [f"{i+1}, {l.strip()}" for i, l in enumerate(m_in.split('\n')) if l.strip()]
            if res: st.code("\n".join(res), language='text')

if __name__ == "__main__":
    main()
