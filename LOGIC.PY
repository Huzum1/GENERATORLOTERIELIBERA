import streamlit as st
import numpy as np
import pandas as pd
import secrets
import time
import io

# ==========================================
# CLASA LOGICĂ (ENGINE) - Inginerii 1-5
# ==========================================
class LotoSystem:
    @staticmethod
    def create_pool(total, fixed_limit, extra_count):
        """Implementare Pasul 8: Pool mixt Fix + Random Rest"""
        fixed_part = set(range(1, fixed_limit + 1))
        remaining = list(set(range(1, total + 1)) - fixed_part)
        
        if extra_count > len(remaining):
            extra_count = len(remaining)
            
        extra_part = secrets.SystemRandom().sample(remaining, extra_count)
        return sorted(list(fixed_part.union(set(extra_part))))

    @staticmethod
    def generate(pool, draw, count, method):
        """Generare masivă cu 5 metode (Pasul 4 & 2)"""
        results = []
        rng = np.random.default_rng()
        
        # Optimizare performanță: Pre-generare pentru viteză
        for i in range(1, count + 1):
            if method == "Criptografic (High Security)":
                variant = sorted(secrets.SystemRandom().sample(pool, draw))
            elif method == "Shuffle Optimization":
                p_copy = list(pool)
                rng.shuffle(p_copy)
                variant = sorted(p_copy[:draw])
            elif method == "Uniform Distribution":
                variant = sorted(rng.choice(pool, size=draw, replace=False))
            elif method == "Step-Wise Logic":
                variant = sorted(rng.choice(pool, size=draw, replace=False, shuffle=True))
            else: # Standard Fast
                variant = sorted(rng.choice(pool, size=draw, replace=False))
            
            # Formatare Pasul 6: ID, Combinatie
            results.append({
                "ID": f"ID_{i}", 
                "Combinatie": " ".join(map(str, variant))
            })
        return pd.DataFrame(results)

# ==========================================
# INTERFAȚĂ (UI/UX) - Inginerii 6-10
# ==========================================
def main():
    st.set_page_config(page_title="Ultra Loto Gen", layout="wide")
    
    # CSS pentru scroll și estetică (Pasul 6)
    st.markdown("""
        <style>
        .main { background-color: #f5f7f9; }
        .stDataFrame { border: 2px solid #4CAF50; border-radius: 10px; }
        </style>
    """, unsafe_allow_html=True)

    st.title("🎰 Generator Loto Universal - Pro Level")
    st.info("Sistem optimizat pentru generări de peste 15.000 de variante fără lag.")

    # SIDEBAR - CONFIGURARE (Pasul 3 & 8)
    with st.sidebar:
        st.header("⚙️ Configurare")
        total_balls = st.number_input("Total numere în urnă", 1, 500, 80)
        draw_size = st.number_input("Numere extrase per variantă", 1, total_balls, 12)
        
        st.divider()
        st.subheader("🛠️ Construcție Pool (Pasul 8)")
        f_limit = st.number_input("Interval fix (1 la X):", 1, total_balls, 25)
        e_count = st.number_input("Adaugă random din restul urnei:", 0, total_balls-f_limit, 15)
        
        st.divider()
        v_count = st.number_input("Număr variante de generat", 1, 100000, 15000)
        gen_method = st.selectbox("Metodă Generare (Pasul 4)", 
                                 ["Standard Fast", "Criptografic (High Security)", 
                                  "Shuffle Optimization", "Uniform Distribution", "Step-Wise Logic"])

    # ZONE DE LUCRU (Tab-uri)
    tab_auto, tab_manual = st.tabs(["⚡ Generare Automată", "📝 Introducere Manuală"])

    with tab_auto:
        col_btn, col_stats = st.columns([1, 2])
        
        if col_btn.button("LANSEAZĂ GENERAREA", use_container_width=True):
            start = time.time()
            
            # Execuție
            current_pool = LotoSystem.create_pool(total_balls, f_limit, e_count)
            df = LotoSystem.generate(current_pool, draw_size, v_count, gen_method)
            
            durata = time.time() - start
            col_stats.success(f"Finalizat! {v_count} variante în {durata:.3f} secunde.")

            # Vizualizare cu Scroll (Pasul 6)
            st.subheader("Primele 10 variante (Scroll pentru restul)")
            st.dataframe(df, height=450, use_container_width=True)

            # Export TXT (Pasul 7)
            # Pregătire format rânduri: 1, 2, 3... (Pasul 5)
            output_text = ""
            for idx, row in df.iterrows():
                output_text += f"{idx + 1}, {row['Combinatie']}\n"
            
            st.download_button(
                label="📥 DESCARCĂ REZULTATE (.TXT)",
                data=output_text,
                file_name="extragere_loto.txt",
                mime="text/plain",
                use_container_width=True
            )

    with tab_manual:
        st.subheader("Adăugare variante manual")
        manual_in = st.text_area("Introdu variantele (câte una pe rând):", height=250, placeholder="Exemplu:\n10 20 30 40\n11 22 33 44")
        
        if st.button("Procesează datele manuale"):
            lines = manual_in.strip().split('\n')
            manual_data = []
            for i, line in enumerate(lines):
                if line.strip():
                    manual_data.append({"ID": f"MAN-{i+1}", "Combinatie": line.strip()})
            
            if manual_data:
                df_man = pd.DataFrame(manual_data)
                st.table(df_man.head(10))
                
                # Export pentru manual
                man_txt = "\n".join([f"{i+1}, {d['Combinatie']}" for i, d in enumerate(manual_data)])
                st.download_button("Descarcă Manual .TXT", man_txt, "manual.txt", "text/plain")

if __name__ == "__main__":
    main()
