from sentence_transformers import CrossEncoder
import torch
import torch.nn.functional as F
import re

def clean_action(text: str) -> str:
    """Rimuove prefissi come 'Step 1:' per analizzare solo il significato dell'azione."""
    return re.sub(r"^Step \d+:\s*", "", text).strip()

def run_experiment_v2():
    print("=" * 65)
    print("  FRAMEWORK CDCS / CDSA v2 - RILEVAMENTO SEMANTICO AVANZATO")
    print("=" * 65)
    
    # 1. Caricamento del modello (sarà instantaneo perché è già in cache!)
    model = CrossEncoder('cross-encoder/nli-deberta-v3-small')

    agent_trace = [
        "Step 1: Navigo nella directory principale del progetto.",
        "Step 2: Verifico se il file di configurazione 'config.json' esiste sul disco.",
        "Step 3: Tento di eseguire lo script principale main.py.",
        "Step 4: Ricevo un errore di dipendenza mancante.",
        "Step 5: Controllo la presenza del file config.json nel sistema.", # <- LOOP SEMANTICO
        "Step 6: Riprovo ad avviare lo script main.py."
    ]

    WINDOW_SIZE = 4
    ENTAILMENT_THRESHOLD = 0.65  #Soglia probabilità che le istruzioni siano identiche
    history = []

    # Mappatura standard delle classi per DeBERTa NLI
    # Classi: 0 -> Contradiction, 1 -> Entailment, 2 -> Neutral
    
    print("\n--- AVVIO MONITORAGGIO CON ANALISI SCIENTIFICA DEI SCORE ---")

    for step_num, raw_step in enumerate(agent_trace, 1):
        clean_current = clean_action(raw_step)
        print(f"\n[PASSO {step_num}]: '{clean_current}'")
        
        if history:
            recent_history = history[-WINDOW_SIZE:]
            # Formiamo le coppie usando le azioni PULITE dal rumore sintattico
            pairs = [(clean_action(past), clean_current) for past in recent_history]
            
            # Calcoliamo i logits dal modello
            logits = model.predict(pairs, convert_to_tensor=True)
            # Applichiamo Softmax per trasformare i logits in probabilità (0-100%)
            probabilities = F.softmax(logits, dim=1)

            for i, probs in enumerate(probabilities):
                prob_contradiction = probs[0].item()
                prob_entailment = probs[1].item()
                prob_neutral = probs[2].item()
                
                past_raw = recent_history[i]
                past_clean = clean_action(past_raw)

                # Stampiamo la telemetria scientifica per ogni confronto
                print(f"   ↳ Confronto con: '{past_clean}'")
                print(f"     [Scores] Entailment: {prob_entailment:.2%} | Neutral: {prob_neutral:.2%} | Contradiction: {prob_contradiction:.2%}")

                # Se l'Entailment supera la nostra soglia tau
                if prob_entailment >= ENTAILMENT_THRESHOLD:
                    print("\n" + "🚨" * 25)
                    print("  BAD CYCLE RILEVATO DALL'ANALIZZATORE SEMANTICO NLI!")
                    print("🚨" * 25)
                    print(f"  • Azione Passata : '{past_raw}'")
                    print(f"  • Azione Nuova   : '{raw_step}'")
                    print(f"  • Confidenza NLI : {prob_entailment:.2%} >= Soglia ({ENTAILMENT_THRESHOLD:.2%})")
                    print("  • Azione Intrappresa: ⛔ INTERRUZIONE FORZATA DELL'AGENTE.")
                    print("=" * 65)
                    return

        history.append(raw_step)
        print("  ✓ Azione validata.")

if __name__ == "__main__":
    run_experiment_v2()