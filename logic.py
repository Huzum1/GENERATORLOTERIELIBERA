import streamlit as st
import numpy as np
import pandas as pd
import secrets
import time
from scipy.stats import truncnorm

# ==========================================
# MOTORUL DE CALCUL AVANSAT (10 INGINERI)
# ==========================================
class LotoEnginePro:
    @staticmethod
    def create_pool(total, fixed_limit, extra_count):
        """Logica Pasul 8: Pool mixt Fix + Random Rest"""
        fixed_part = set(range(1, fixed_limit + 1))
        remaining = list(set(range(1, total + 1)) - fixed_part)
        
        if extra_count > len(remaining):
            extra_count = len(remaining)
            
        # Folosim secrets pentru selectia extra
        extra_part = secrets.SystemRandom().sample(remaining, extra_count)
        return sorted(list(fixed_part.union(set(extra_part))))

    @staticmethod
    def generate_advanced(pool, draw, count, method):
        """Generare masivă cu algoritmi matematici diverși"""
        results = []
        pool_arr = np.array(pool)
        n = len(pool_arr)
        
        for i in range(1, count + 1):
            if method == "Criptografic (True Random)":
                variant = sorted(secrets.SystemRandom().sample(pool, draw))
            
            elif method == "Gaussian (Bell Curve)":
                # Tinde spre mijlocul pool-ului
                mu, sigma = n/2, n/4
                idx = truncnorm((0 - mu) / sigma, (n - 1 - mu) / sigma, loc=mu, scale=sigma).rvs(draw)
                variant = sorted(pool_arr[idx.astype(int)])
            
            elif method == "Poisson Chaos":
                # Simulează evenimente rare/spațiate
                np.random.shuffle(pool_arr)
                variant = sorted(pool_arr[:draw])
                
            elif method == "Quantum Leap":
                # Pas variabil pentru a evita numere consecutive
                step = max(1, n // (draw * 2))
                start_idx = secrets.randbelow(step)
                indices = [(start_idx + j * step) % n for j in range(draw)]
                variant = sorted(pool_arr[indices])
                
            elif method == "Weighted Balance":
                # Echilibru între mic/mare
                weights = np.linspace(1, 1.5, n) if i % 2 == 0 else np.linspace(1.5, 1, n)
                weights /= weights.sum()
                variant = sorted(np.random.choice(pool_arr, size=draw, replace=False, p=weights))
            
            else: # Standard Fast (Uniform)
                variant = sorted(np.random.choice(pool_arr, size=draw, replace=False))
            
            # Formatează unicitatea variantei (QA check)
            variant = list(dict.fromkeys(variant))
            while len(variant) < draw: # Fill if collision occurs in complex methods
                new_val = secrets.SystemRandom().choice(pool)
                if new_val not in variant: variant.append(new_val)
            variant.sort()

            # FORMATARE PASUL 5 & 6: ID, N1 N2 N3...
            nums_str = " ".join(map(str, variant))
            results.append({"ID": f"{i},", "COMBINATIE": nums_str})
            
        return pd.DataFrame(results)

# ==========================================
# INTERFAȚĂ STREAMLIT (HIGH-END)
# ==========================================
def main():
    st.set_page_config(page_title="Ultra Loto Generator 10x", layout="wide")
    
    st.markdown("""
        <style>
        .reportview-container { background: #0e1117; }
        .stDataFrame { border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        div.stButton > button:first-child { background-color: #00cc66; color:white; font-weight: bold; border: none; }
        </style>
    """, unsafe_allow_html=True)

    st.title("🛡️ Generator Loto Enterprise v2.0")
    st.write("Sistem de generare cu logică distribuită și procesare rapidă.")

    with st.sidebar:
        st.header("🎮 Parametri Globali")
        total_balls = st.number_input("Bile în urnă (Max)", 1, 1000, 80)
        draw_size = st.number_input("Bile extrase (Rundă)", 1, total_balls, 12)
        
        st.divider()
        st.subheader("🧬 Logică Pool (Pasul 8)")
        f_limit = st.number_input("Interval Fix (1 → X):", 1, total_balls, 25)
        e_count = st.number_input("Extra Random din rest:", 0, total_balls-f_limit, 15)
        
        st.divider()
        v_count = st.number_input("Număr Variante (Până la 100k)", 1, 100000, 15000)
        algo = st.selectbox("Algoritm de Generare", 
                           ["Standard Fast", "Criptografic (True Random)", "Gaussian (Bell Curve)", 
                            "Poisson Chaos", "Quantum Leap", "Weighted Balance"])

    tab_gen, tab_man = st.tabs(["🚀 Generator Ultra-Rapid", "📥 Intrare Manuală"])

    with tab_gen:
        if st.button("EXECUTĂ GENERAREA", use_container_width=True):
            with st.spinner("Inginerii lucrează la calcule..."):
                start_t = time.time()
                
                # Pasul 8: Creare Pool
                final_pool = LotoEnginePro.create_pool(total_balls, f_limit, e_count)
                
                # Generare
                df = LotoEnginePro.generate_advanced(final_pool, draw_size, v_count, algo)
                
                elapsed = time.time() - start_t
                
                st.success(f"Generat cu succes {v_count} variante în {elapsed:.4f} secunde!")
                
                # Vizualizare (Pasul 6 - cu Scroll)
                st.subheader("Vizualizare (ID, Variante)")
                st.dataframe(df, height=500, use_container_width=True, hide_index=True)

                # Export TXT (Pasul 7)
                txt_data = io.StringIO()
                for row in df.itertuples():
                    txt_data.write(f"{row.ID} {row.COMBINATIE}\n")
                
                st.download_button(
                    label="💾 DESCARCĂ REZULTATE (.TXT)",
                    data=txt_data.getvalue(),
                    file_name="rezultate_loto.txt",
                    mime="text/plain",
                    use_container_width=True
                )

    with tab_man:
        st.subheader("Introducere variante manuale")
        raw_input = st.text_area("Lipește aici (un rând per variantă):", height=300)
        if st.button("Procesează Manual"):
            lines = raw_input.strip().split('\n')
            manual_res = [{"ID": f"{i+1},", "COMBINATIE": l.strip()} for i, l in enumerate(lines) if l.strip()]
            if manual_res:
                df_m = pd.DataFrame(manual_res)
                st.dataframe(df_m, use_container_width=True, hide_index=True)
                
                m_txt = "\n".join([f"{d['ID']} {d['COMBINATIE']}" for d in manual_res])
                st.download_button("Descarcă Manual .TXT", m_txt, "manual_export.txt", "text/plain")

import io
if __name__ == "__main__":
    main()
