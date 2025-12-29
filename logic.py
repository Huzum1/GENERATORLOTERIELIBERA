import streamlit as st
import numpy as np
import pandas as pd
import secrets
import time

# ==========================================
# MOTORUL DE CALCUL (ENGINE)
# ==========================================
class LotoSystem:
    @staticmethod
    def create_pool(total, fixed_limit, extra_count):
        """Logica Pasul 8: Pool mixt Fix + Random Rest"""
        fixed_part = set(range(1, fixed_limit + 1))
        remaining = list(set(range(1, total + 1)) - fixed_part)
        
        if extra_count > len(remaining):
            extra_count = len(remaining)
            
        extra_part = secrets.SystemRandom().sample(remaining, extra_count)
        return sorted(list(fixed_part.union(set(extra_part))))

    @staticmethod
    def generate(pool, draw, count, method):
        """Generare masivă cu formatare strictă: ID, Combinatie (Spațiu între numere)"""
        results = []
        rng = np.random.default_rng()
        
        for i in range(1, count + 1):
            if method == "Criptografic (High Security)":
                variant = sorted(secrets.SystemRandom().sample(pool, draw))
            elif method == "Shuffle Optimization":
                p_copy = list(pool)
                rng.shuffle(p_copy)
                variant = sorted(p_copy[:draw])
            elif method == "Uniform Distribution":
                variant = sorted(rng.choice(pool, size=draw, replace=False))
            else:
                variant = sorted(rng.choice(pool, size=draw, replace=False))
            
            # FORMATARE CERUTĂ: ID, Combinatie (Ex: 1, 43 45 21 24)
            # Folosim spațiu între numere conform cerinței noi
            nums_joined = " ".join(map(str, variant))
            results.append({
                "ID_RUNDA": f"{i},", 
                "COMBINATIE": nums_joined
            })
        return pd.DataFrame(results)

# ==========================================
# INTERFAȚĂ UTILIZATOR (UI/UX)
# ==========================================
def main():
    st.set_page_config(page_title="Ultra Loto Gen", layout="wide")
    
    # CSS pentru aspect profesional și scroll (Pasul 6)
    st.markdown("""
        <style>
        .stDataFrame { border: 1px solid #e6e9ef; border-radius: 5px; }
        [data-testid="stMetricValue"] { font-size: 24px; }
        </style>
    """, unsafe_allow_html=True)

    st.title("🎰 Generator Loto Universal - Mod Fix/Random")

    # SIDEBAR - CONFIGURARE
    with st.sidebar:
        st.header("⚙️ Setări Sistem")
        total_balls = st.number_input("Total numere în urnă", 1, 500, 80)
        draw_size = st.number_input("Numere extrase per variantă", 1, total_balls, 12)
        
        st.divider()
        st.subheader("🛠️ Configurare Pool (Pasul 8)")
        f_limit = st.number_input("Interval fix (1 la X):", 1, total_balls, 25)
        e_count = st.number_input("Adaugă random din rest:", 0, total_balls-f_limit, 15)
        
        st.divider()
        v_count = st.number_input("Variante de generat", 1, 100000, 15000)
        gen_method = st.selectbox("Metodă Generare", 
                                 ["Standard Fast", "Criptografic (High Security)", 
                                  "Shuffle Optimization", "Uniform Distribution"])

    tab_auto, tab_manual = st.tabs(["⚡ Generare Automată", "📝 Introducere Manuală"])

    with tab_auto:
        if st.button("LANSEAZĂ GENERAREA", use_container_width=True):
            start = time.time()
            
            current_pool = LotoSystem.create_pool(total_balls, f_limit, e_count)
            df = LotoSystem.generate(current_pool, draw_size, v_count, gen_method)
            
            durata = time.time() - start
            st.success(f"Finalizat! {v_count} variante în {durata:.3f} secunde.")

            # Vizualizare cu Scroll (Pasul 6)
            st.subheader("Vizualizare Variante (ID, Combinatie)")
            # Afișăm dataframe-ul optimizat
            st.dataframe(df, height=500, use_container_width=True, hide_index=True)

            # Export TXT (Pasul 7) - Format: ID, N1 N2 N3...
            txt_output = ""
            for row in df.itertuples():
                txt_output += f"{row.ID_RUNDA} {row.COMBINATIE}\n"
            
            st.download_button(
                label="📥 DESCARCĂ REZULTATE (.TXT)",
                data=txt_output,
                file_name="extragere_loto.txt",
                mime="text/plain",
                use_container_width=True
            )

    with tab_manual:
        st.subheader("Adăugare manuală")
        manual_in = st.text_area("Introdu numerele (Ex: 10 20 30...):", height=200)
        
        if st.button("Procesează manual"):
            lines = manual_in.strip().split('\n')
            man_list = []
            for i, line in enumerate(lines):
                if line.strip():
                    man_list.append({
                        "ID_RUNDA": f"{i+1},", 
                        "COMBINATIE": line.strip()
                    })
            if man_list:
                df_man = pd.DataFrame(man_list)
                st.dataframe(df_man, use_container_width=True, hide_index=True)
                
                # Export manual
                man_txt = "\n".join([f"{d['ID_RUNDA']} {d['COMBINATIE']}" for d in man_list])
                st.download_button("Descarcă Manual .TXT", man_txt, "manual.txt", "text/plain")

if __name__ == "__main__":
    main()
