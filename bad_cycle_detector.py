"""
Framework di rilevamento "Bad Cycles" tramite Cross-Encoder NLI.
Esperimento per la validazione semantica contro i loop inutili negli agenti LLM.
"""

from sentence_transformers import CrossEncoder
import time

def run_experiment():
    print("=" * 60)
    print("  FRAMEWORK CDCS / CDSA - RILEVAMENTO BAD CYCLES NLI")
    print("=" * 60)
    
    print("\n[1/3] Caricamento del modello Cross-Encoder NLI...")
    start_time = time.time()
    
    # Modello Cross-Encoder addestrato su NLI (DeBERTa-v3 small: rapido ed eccellente nella semantica)
    # Riconosce le classi: Contradiction, Entailment, Neutral
    model = CrossEncoder('cross-encoder/nli-deberta-v3-small')
    print(f"Modello caricato in {time.time() - start_time:.2f} secondi!\n")

    # Simulazione della traccia di azioni dell'agente LLM.
    # Nota come lo Step 2 e lo Step 5 dicano la STESSA identica cosa con parole diverse!
    # Un approccio a distanza di Hamming o sintattico fallirebbe, l'NLI invece la sgama.
    agent_trace = [
        "Step 1: Navigo nella directory principale del progetto.",
        "Step 2: Verifico se il file di configurazione 'config.json' esiste sul disco.",
        "Step 3: Tento di eseguire lo script principale main.py.",
        "Step 4: Ricevo un errore di dipendenza mancante.",
        "Step 5: Controllo la presenza del file config.json nel sistema.", # <- LOOP SEMANTICO CON STEP 2!
        "Step 6: Riprovo ad avviare lo script main.py."
    ]

    # Dimensione della sliding window temporale (CDCS)
    WINDOW_SIZE = 3
    history = []

    print("[2/3] Avvio simulazione agente e monitoraggio semantico (CDSA)...")
    print("-" * 60)

    for step_num, current_step in enumerate(agent_trace, 1):
        print(f"\n[AGENTE - PASSO {step_num}]: '{current_step}'")
        
        # Valutazione semantica rispetto alla finestra temporale (CDSA)
        if history:
            # Finestra scorrevole degli ultimi N elementi
            recent_history = history[-WINDOW_SIZE:]
            
            # Formiamo le coppie (azione_passata, azione_corrente) per l'NLI
            pairs = [(past_step, current_step) for past_step in recent_history]
            
            # Predizione dei punteggi logits
            scores = model.predict(pairs)
            
            # Mapping delle classi NLI di cross-encoder/nli-deberta-v3-small
            # 0: Contradiction, 1: Entailment, 2: Neutral
            label_mapping = ['Contradiction', 'Entailment', 'Neutral']
            
            for i, score in enumerate(scores):
                pred_index = score.argmax()
                predicted_label = label_mapping[pred_index]
                past_action = recent_history[i]
                
                # Se il modello rileva ENTAILMENT (ridondanza/ripetizione semantica)
                if predicted_label == 'Entailment':
                    print("\n" + "🚨" * 25)
                    print("  BAD CYCLE RILEVATO DALL'ANALIZZATORE SEMANTICO NLI!")
                    print("🚨" * 25)
                    print(f"  • Azione Passata (Premessa) : '{past_action}'")
                    print(f"  • Azione Nuova (Ipotesi)   : '{current_step}'")
                    print(f"  • Relazione Semantica     : ENTAILMENT (Ripetizione/Implicazione Inutile)")
                    print("  • Azione Intrappresa       : ⛔ INTERRUZIONE FORZATA (Loop Evitato)")
                    print("=" * 60)
                    return

        # Aggiungiamo l'azione allo storico se il controllo passa
        history.append(current_step)
        print("  ✓ Azione validata. Nessun ciclo inutile rilevato.")

    print("\n[3/3] Simulazione completata senza interruzioni.")

if __name__ == "__main__":
    run_experiment()