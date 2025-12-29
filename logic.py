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
        elif strategy == "Entropy Shuffle":
            px = p.copy()
            for _ in range(5): rng.shuffle(px)
            variant = px[:draw]
        elif strategy == "Delta Distance":
            variant = sorted(rng.choice(p, size=draw, replace=False))
        else: # Default
            variant = rng.choice(p, size=draw, replace=False)

        final_v = sorted(list(set(variant)))
        while len(final_v) < draw:
            extra = secrets.SystemRandom().choice(p)
            if extra not in final_v: final_v.append(extra)
        return sorted(final_v)

# ==========================================
# INTERFATA UTILIZATOR (UI/UX)
# ==========================================
def main():
    st.set_page_config(page_title="Ultra Loto Master", layout="wide")
    
    # CSS Custom pentru a forța înălțimea ferestrei de copy
    st.markdown("""
        <style>
        .stCodeBlockContainer {
            max-height: 400px;
            overflow-y: auto;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🛡️ Generator Loto Profesional - All-in-One Copy")

    all_strategies = ["Criptografic", "Gaussian", "Quantum Leap", "Prime Affinity", "Entropy Shuffle", "Delta Distance"]

    with st.sidebar:
        st.header("⚙️ Setări")
        total = st.number_input("Bile în urnă", 1, 1000, 80)
        draw = st.number_input("Numere extrase", 1, total, 12)
        
        st.divider()
        st.subheader("🧬 Pool (Pasul 8)")
        f_lim = st.number_input("Interval Fix (1-X):", 1, total, 25)
        e_cnt = st.number_input("Extra Random:", 0, total-f_lim, 15)
        
        st.divider()
        v_count = st.number_input("Variante de generat", 1, 100000, 15000)
        
        st.subheader("🎯 Mix Strategii")
        selected_strats = [s for s in all_strategies if st.checkbox(s, value=(s == "Criptografic"))]

    if not selected_strats:
        st.warning("⚠️ Bifează cel puțin o strategie!")
        return

    tab_gen, tab_man = st.tabs(["🚀 Generator Principal", "📥 Mod Manual"])

    with tab_gen:
        if st.button("LANCEAZĂ GENERAREA COMPLETĂ", use_container_width=True):
            t1 = time.time()
            pool = LotoHybridEngine.create_pool(total, f_lim, e_cnt)
            pool_arr = np.array(pool)
            
            raw_variants = []
            per_strat = v_count // len(selected_strats)
            
            # Generare pe strategii
            for s_name in selected_strats:
                for _ in range(per_strat):
                    v = LotoHybridEngine.generate_single_variant(pool_arr, draw, s_name)
                    raw_variants.append(" ".join(map(str, v)))
            
            # Completare până la v_count
            while len(raw_variants) < v_count:
                v = LotoHybridEngine.generate_single_variant(pool_arr, draw, selected_strats[0])
                raw_variants.append(" ".join(map(str, v)))
            
            # Mixare finală
            secrets.SystemRandom().shuffle(raw_variants)
            
            # Pregătire text pentru Copy (toate variantele)
            full_text_io = io.StringIO()
            for i, variant in enumerate(raw_variants):
                full_text_io.write(f"{i+1}, {variant}\n")
            
            full_output = full_text_io.getvalue()
            t2 = time.time()
            
            st.success(f"Generat {v_count} variante în {t2-t1:.3f} secunde!")

            # --- SECȚIUNEA DE COPY-ALL CU SCROLL ---
            st.subheader("📋 Preview Complet (Buton Copy sus-dreapta pentru TOATE)")
            st.info("Fereastra de mai jos conține toate variantele. Folosește butonul de copy din colț pentru a le lua pe toate odată.")
            st.code(full_output, language='text') 
            # ----------------------------------------

            # Export TXT (pentru siguranță)
            st.download_button("📥 DESCARCĂ FIȘIER .TXT", full_output, "loto_export.txt", "text/plain", use_container_width=True)

    with tab_man:
        manual_in = st.text_area("Introdu date manual:", height=300)
        if st.button("Procesează Manual"):
            res = [f"{i+1}, {l.strip()}" for i, l in enumerate(manual_in.split('\n')) if l.strip()]
            if res:
                st.code("\n".join(res), language='text')

if __name__ == "__main__":
    main()
