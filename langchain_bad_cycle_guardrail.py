"""
Integrazione Framework CDCS/CDSA con LangChain.
Versione 3.0: Dual-Condition Detection (Entailment + High Neutrality).
"""

from typing import Any, Dict, List
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction
from sentence_transformers import CrossEncoder
import torch.nn.functional as F

class BadCycleDetectedError(Exception):
    """Eccezione personalizzata sollevata quando viene rilevato un loop semantico inutile."""
    pass

class NLIBadCycleGuardrail(BaseCallbackHandler):
    def __init__(self, window_size: int = 4, threshold: float = 0.60):
        super().__init__()
        print("\n[Guardrail Init] Inizializzazione Cross-Encoder NLI per LangChain...")
        self.model = CrossEncoder('cross-encoder/nli-deberta-v3-small')
        self.window_size = window_size
        self.threshold = threshold
        self.history: List[Dict[str, str]] = []

    def _extract_intent(self, action: AgentAction) -> str:
        """Estrae l'intento semantico puro ignorando il nome del tool."""
        return str(action.tool_input).strip()

    def check_semantic_loop(self, action: AgentAction):
        current_intent = self._extract_intent(action)

        if self.history:
            recent_history = self.history[-self.window_size:]
            pairs = [(past['intent'], current_intent) for past in recent_history]
            
            logits = self.model.predict(pairs, convert_to_tensor=True)
            probabilities = F.softmax(logits, dim=1)

            for i, probs in enumerate(probabilities):
                prob_contradiction = probs[0].item()
                prob_entailment = probs[1].item()
                prob_neutral = probs[2].item()
                
                past_item = recent_history[i]

                print(f"   ↳ [NLI Test] vs '{past_item['intent']}'")
                print(f"     Scores -> Entailment: {prob_entailment:.2%} | Neutral: {prob_neutral:.2%} | Contradiction: {prob_contradiction:.2%}")

                # LOGICA DUAL-CONDITION RILEVAMENTO LOOP
                is_entailment = prob_entailment >= self.threshold
                is_redundant_neutral = (prob_neutral > 0.85) and (prob_contradiction < 0.02)

                if is_entailment or is_redundant_neutral:
                    reason = "Entailment Semantico" if is_entailment else "Ridondanza Cognitiva (High Neutral)"
                    error_msg = (
                        f"\n🚨 BAD CYCLE BLOCCATO DA LANGCHAIN GUARDRAIL 🚨\n"
                        f"• Motivo Rilevamento: {reason}\n"
                        f"• Azione Passata    : '{past_item['raw_description']}'\n"
                        f"• Tentativo Attuale : 'Tool: {action.tool} | Input: {action.tool_input}'\n"
                        f"• Confidenza NLI    : Neutral={prob_neutral:.2%}, Entailment={prob_entailment:.2%}"
                    )
                    raise BadCycleDetectedError(error_msg)

        self.history.append({
            'intent': current_intent,
            'raw_description': f"Tool: {action.tool} | Input: {action.tool_input}"
        })

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Callback invocata da LangChain."""
        print(f"\n[LangChain Callback] Intercettata azione: Tool='{action.tool}', Input='{action.tool_input}'")
        self.check_semantic_loop(action)
        print("  ✓ Azione approvata dal Guardrail NLI.")


# --- TEST SIMULATO DEL WORKFLOW LANGCHAIN ---
def simulate_langchain_execution():
    print("=" * 70)
    print("  TEST DI INTEGRABILITÀ v3: DUAL-CONDITION GUARDRAIL (CDCS/CDSA)")
    print("=" * 70)

    guardrail = NLIBadCycleGuardrail(window_size=3, threshold=0.60)

    simulated_actions = [
        AgentAction(tool="file_search", tool_input="verifico la presenza del file config.json", log=""),
        AgentAction(tool="execute_script", tool_input="main.py", log=""),
        AgentAction(tool="check_system", tool_input="controllo se il file config.json esiste nel sistema", log=""), # <- LOOP
    ]

    print("\n--- INIZIO ESECUZIONE FLUSSO AGENTE ---")
    
    try:
        for step, action in enumerate(simulated_actions, 1):
            guardrail.on_agent_action(action)
            print(f"  --> [Step {step}] Esecuzione reale del tool '{action.tool}' in corso...")
            
    except BadCycleDetectedError as e:
        print(e)
        print("\n⛔ Workflow LangChain interrotto con successo prima dello spreco di token!")
    else:
        print("\nWorkflow completato senza interruzioni.")

if __name__ == "__main__":
    simulate_langchain_execution()