# logic_engine.py
"""A tiny propositional Knowledge Base with a forward-chaining inference engine.

Facts are plain strings (e.g. "TargetVisible"). Rules are Horn clauses stored
as (premises, conclusion) tuples, read as: p1 AND p2 AND ... => conclusion.
"""


class KnowledgeBase:
    def __init__(self):
        self.facts = set()   # Unique facts currently known to be true
        self.rules = []      # Horn clauses: ([premise, ...], conclusion)

    def tell_fact(self, fact):
        """Assert a single fact (a percept) into the KB."""
        self.facts.add(fact)

    def tell_rule(self, premises, conclusion):
        """Add a Horn clause: all premises together imply the conclusion."""
        self.rules.append((list(premises), conclusion))

    def clear_facts(self):
        """Forget every fact but keep the rules (new percepts, same domain)."""
        self.facts.clear()

    def forward_chain(self):
        """Data-driven inference: keep firing rules whose premises are all
        known (Modus Ponens) until a full pass adds no new facts."""
        new_facts_added = True

        while new_facts_added:
            new_facts_added = False

            for premises, conclusion in self.rules:
                if conclusion not in self.facts:
                    # Modus Ponens: if every premise holds, so does the conclusion
                    if all(p in self.facts for p in premises):
                        self.facts.add(conclusion)
                        new_facts_added = True

        return self.facts
